#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t LLAMA_SEQ_STATE_MAGIC = 0xaf143cd8;
static constexpr float    FUSION_ALPHA          = 0.5f;
static constexpr int32_t  SEQ_A                 = 0;
static constexpr int32_t  SEQ_B                 = 1;
static constexpr int32_t  SEQ_FUSED             = 2;

struct app_params {
    std::string model_path;
    std::string prompt_a = "CacheBlend prompt A: local knowledge anchor for edge inference.";
    std::string prompt_b = "CacheBlend prompt B: retrieved context shard for static fusion.";
    int32_t n_tokens = 32;
    int32_t n_gpu_layers = 99;
    int32_t n_verify = 2;
};

enum class payload_kind {
    key,
    value,
};

struct payload_range {
    payload_kind kind;
    uint32_t stream;
    uint32_t layer;
    int32_t  type;
    size_t   offset;
    size_t   nbytes;
};

struct state_layout {
    uint32_t n_stream = 0;
    std::vector<uint32_t> cell_counts;
    std::vector<payload_range> payloads;
};

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t n_seq_max)
        : batch(llama_batch_init(n_tokens, 0, n_seq_max)) {}

    ~llama_batch_ptr() {
        llama_batch_free(batch);
    }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
};

struct byte_reader {
    const std::vector<uint8_t> & data;
    size_t off = 0;

    template <typename T>
    T read() {
        ensure_available(sizeof(T));
        T value;
        std::memcpy(&value, data.data() + off, sizeof(T));
        off += sizeof(T);
        return value;
    }

    void skip(size_t n) {
        ensure_available(n);
        off += n;
    }

    void ensure_available(size_t n) const {
        if (off > data.size() || n > data.size() - off) {
            throw std::runtime_error("state buffer is truncated");
        }
    }
};

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "\nusage:\n"
        "  %s -m model.gguf [--n-tokens N] [-ngl N] [--prompt-a TEXT] [--prompt-b TEXT] [--n-verify N]\n\n"
        "example:\n"
        "  %s -m models/qwen2-7b-instruct-q4_k_m.gguf --n-tokens 32 -ngl 99 --n-verify 2\n\n",
        argv0, argv0);
}

static bool parse_args(int argc, char ** argv, app_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                print_usage(argv[0]);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-m") {
            const char * value = require_value("-m");
            if (!value) {
                return false;
            }
            params.model_path = value;
        } else if (arg == "-ngl") {
            const char * value = require_value("-ngl");
            if (!value) {
                return false;
            }
            params.n_gpu_layers = std::stoi(value);
        } else if (arg == "--n-tokens") {
            const char * value = require_value("--n-tokens");
            if (!value) {
                return false;
            }
            params.n_tokens = std::stoi(value);
        } else if (arg == "--n-verify") {
            const char * value = require_value("--n-verify");
            if (!value) {
                return false;
            }
            params.n_verify = std::stoi(value);
        } else if (arg == "--prompt-a") {
            const char * value = require_value("--prompt-a");
            if (!value) {
                return false;
            }
            params.prompt_a = value;
        } else if (arg == "--prompt-b") {
            const char * value = require_value("--prompt-b");
            if (!value) {
                return false;
            }
            params.prompt_b = value;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (params.model_path.empty() || params.n_tokens <= 0 || params.n_verify <= 0) {
        print_usage(argv[0]);
        return false;
    }

    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int32_t n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n <= 0) {
        throw std::runtime_error("failed to estimate token count");
    }

    std::vector<llama_token> tokens(n);
    const int32_t n_actual = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    if (n_actual < 0) {
        throw std::runtime_error("failed to tokenize prompt");
    }

    tokens.resize(n_actual);
    return tokens;
}

static std::vector<llama_token> make_exact_length_tokens(
        const llama_vocab * vocab,
        const std::string & prompt,
        int32_t n_tokens,
        const char * label) {
    std::vector<llama_token> tokens = tokenize(vocab, prompt);

    if (tokens.empty()) {
        tokens.push_back(llama_vocab_bos(vocab));
    }

    if ((int32_t) tokens.size() > n_tokens) {
        tokens.resize(n_tokens);
    }

    // Level 1 PoC only cares that KV_1 and KV_2 have exactly the same cell count.
    // If a prompt is too short, repeat its last token to construct an artificial
    // equal-length sequence. Later CacheBlend work should replace this with real
    // shard slicing / prefix alignment instead of token padding.
    while ((int32_t) tokens.size() < n_tokens) {
        tokens.push_back(tokens.back());
    }

    std::fprintf(stderr, "%s exact token count = %d\n", label, (int) tokens.size());
    return tokens;
}

static std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return "<piece-error>";
    }
    return std::string(buf, n);
}

static bool decode_tokens_to_seq(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_seq_id seq_id,
        int32_t pos0 = 0) {
    llama_batch_ptr holder(tokens.size(), 1);
    llama_batch & batch = holder.batch;
    batch.n_tokens = (int32_t) tokens.size();

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i]    = (i == batch.n_tokens - 1) ? 1 : 0;
    }

    const int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        std::fprintf(stderr, "llama_decode failed for seq_id=%d, ret=%d\n", seq_id, ret);
        return false;
    }

    return true;
}

static bool decode_one_token(
        llama_context * ctx,
        llama_token token,
        llama_pos pos,
        llama_seq_id seq_id) {
    llama_batch_ptr holder(1, 1);
    llama_batch & batch = holder.batch;
    batch.n_tokens     = 1;
    batch.token[0]     = token;
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = seq_id;
    batch.logits[0]    = 1;

    const int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        std::fprintf(stderr, "llama_decode failed for verification seq_id=%d, pos=%d, ret=%d\n", seq_id, pos, ret);
        return false;
    }

    return true;
}

static std::vector<uint8_t> export_seq_state_host(llama_context * ctx, llama_seq_id seq_id) {
    const size_t state_size = llama_state_seq_get_size(ctx, seq_id);
    if (state_size == 0) {
        throw std::runtime_error("llama_state_seq_get_size returned 0");
    }

    std::vector<uint8_t> buffer(state_size);
    const size_t n_written = llama_state_seq_get_data(ctx, buffer.data(), buffer.size(), seq_id);
    if (n_written != buffer.size()) {
        throw std::runtime_error("llama_state_seq_get_data wrote an unexpected number of bytes");
    }

    return buffer;
}

static state_layout parse_state_layout(const std::vector<uint8_t> & state, const char * label) {
    byte_reader r{state};
    state_layout layout;

    const uint32_t magic = r.read<uint32_t>();
    const int32_t src_seq_id = r.read<int32_t>();
    if (magic != LLAMA_SEQ_STATE_MAGIC) {
        throw std::runtime_error("unexpected sequence state magic");
    }

    std::fprintf(stderr, "\n[%s] state bytes = %zu, magic = 0x%08x, src_seq_id = %d\n",
            label, state.size(), magic, src_seq_id);

    layout.n_stream = r.read<uint32_t>();
    layout.cell_counts.reserve(layout.n_stream);
    std::fprintf(stderr, "[%s] n_stream = %u\n", label, layout.n_stream);

    for (uint32_t s = 0; s < layout.n_stream; ++s) {
        const uint32_t cell_count = r.read<uint32_t>();
        layout.cell_counts.push_back(cell_count);
        std::fprintf(stderr, "[%s] stream %u cell_count = %u\n", label, s, cell_count);

        if (cell_count == 0) {
            continue;
        }

        // This parser targets text LLMs such as Qwen2 where n_pos_per_embd == 1.
        // Multimodal M-RoPE states add llama_kv_cell_ext to each cell metadata and
        // need a richer parser before fusion is allowed.
        for (uint32_t i = 0; i < cell_count; ++i) {
            const int32_t  pos      = r.read<int32_t>();
            const uint32_t n_seq_id = r.read<uint32_t>();
            r.skip(sizeof(int32_t) * n_seq_id);
            if (i == 0 || i + 1 == cell_count) {
                std::fprintf(stderr, "[%s]   meta cell[%u] pos = %d, n_seq_id = %u\n",
                        label, i, pos, n_seq_id);
            }
        }

        const uint32_t v_trans = r.read<uint32_t>();
        const uint32_t n_layer = r.read<uint32_t>();
        std::fprintf(stderr, "[%s] data header offset passed, v_trans = %u, n_layer = %u\n",
                label, v_trans, n_layer);

        for (uint32_t il = 0; il < n_layer; ++il) {
            const int32_t  k_type     = r.read<int32_t>();
            const uint64_t k_size_row = r.read<uint64_t>();
            const size_t   k_offset   = r.off;
            const size_t   k_nbytes   = (size_t) cell_count * (size_t) k_size_row;

            // Static fusion hook:
            // [k_offset, k_offset + k_nbytes) is the serialized K payload for this
            // layer. The fusion kernel below type-casts exactly this region and
            // writes 0.5 * K_A + 0.5 * K_B into fused_buffer at the same offset.
            layout.payloads.push_back({ payload_kind::key, s, il, k_type, k_offset, k_nbytes });
            std::fprintf(stderr,
                    "[%s]   K layer %u: type=%d payload_offset=%zu payload_bytes=%zu\n",
                    label, il, k_type, k_offset, k_nbytes);
            r.skip(k_nbytes);
        }

        for (uint32_t il = 0; il < n_layer; ++il) {
            const int32_t v_type = r.read<int32_t>();

            if (!v_trans) {
                const uint64_t v_size_row = r.read<uint64_t>();
                const size_t   v_offset   = r.off;
                const size_t   v_nbytes   = (size_t) cell_count * (size_t) v_size_row;

                // Static fusion hook:
                // [v_offset, v_offset + v_nbytes) is the serialized non-transposed
                // V payload for this layer.
                layout.payloads.push_back({ payload_kind::value, s, il, v_type, v_offset, v_nbytes });
                std::fprintf(stderr,
                        "[%s]   V layer %u: type=%d payload_offset=%zu payload_bytes=%zu\n",
                        label, il, v_type, v_offset, v_nbytes);
                r.skip(v_nbytes);
            } else {
                const uint32_t v_size_el    = r.read<uint32_t>();
                const uint32_t n_embd_v_gqa = r.read<uint32_t>();
                const size_t   v_offset     = r.off;
                const size_t   v_nbytes     = (size_t) cell_count * (size_t) v_size_el * (size_t) n_embd_v_gqa;

                // Static fusion hook:
                // [v_offset, v_offset + v_nbytes) is the serialized transposed V
                // payload. It is stored as n_embd_v_gqa stripes of cell_count
                // elements, but the serialized bytes are contiguous and can be
                // fused element-wise if state A/B layouts are identical.
                layout.payloads.push_back({ payload_kind::value, s, il, v_type, v_offset, v_nbytes });
                std::fprintf(stderr,
                        "[%s]   V layer %u: type=%d transposed payload_offset=%zu payload_bytes=%zu n_embd_v_gqa=%u\n",
                        label, il, v_type, v_offset, v_nbytes, n_embd_v_gqa);
                r.skip(v_nbytes);
            }
        }
    }

    if (r.off != state.size()) {
        throw std::runtime_error("state layout parser did not consume the full buffer");
    }

    return layout;
}

static void require_same_layout(const state_layout & a, const state_layout & b) {
    if (a.n_stream != b.n_stream || a.cell_counts != b.cell_counts || a.payloads.size() != b.payloads.size()) {
        throw std::runtime_error("state layouts are not structurally identical");
    }

    for (size_t i = 0; i < a.payloads.size(); ++i) {
        const payload_range & ra = a.payloads[i];
        const payload_range & rb = b.payloads[i];

        if (ra.kind   != rb.kind   ||
            ra.stream != rb.stream ||
            ra.layer  != rb.layer  ||
            ra.type   != rb.type   ||
            ra.offset != rb.offset ||
            ra.nbytes != rb.nbytes) {
            throw std::runtime_error("payload layout mismatch between state A and state B");
        }
    }
}

static void ensure_range(const std::vector<uint8_t> & buf, const payload_range & r) {
    if (r.offset > buf.size() || r.nbytes > buf.size() - r.offset) {
        throw std::runtime_error("payload range is out of bounds");
    }
}

template <typename T>
static const T * checked_ptr_const(const std::vector<uint8_t> & buf, const payload_range & r) {
    ensure_range(buf, r);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(buf.data() + r.offset);
    if (addr % alignof(T) != 0) {
        throw std::runtime_error("payload pointer is not aligned for requested type");
    }
    return reinterpret_cast<const T *>(buf.data() + r.offset);
}

template <typename T>
static T * checked_ptr_mut(std::vector<uint8_t> & buf, const payload_range & r) {
    ensure_range(buf, r);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(buf.data() + r.offset);
    if (addr % alignof(T) != 0) {
        throw std::runtime_error("payload pointer is not aligned for requested type");
    }
    return reinterpret_cast<T *>(buf.data() + r.offset);
}

static void fuse_f32_payload(
        std::vector<uint8_t> & fused,
        const std::vector<uint8_t> & a,
        const std::vector<uint8_t> & b,
        const payload_range & r) {
    if (r.nbytes % sizeof(float) != 0) {
        throw std::runtime_error("F32 payload byte count is not divisible by sizeof(float)");
    }

    const size_t n = r.nbytes / sizeof(float);
    const float * pa = checked_ptr_const<float>(a, r);
    const float * pb = checked_ptr_const<float>(b, r);
    float * pd = checked_ptr_mut<float>(fused, r);

    for (size_t i = 0; i < n; ++i) {
        pd[i] = FUSION_ALPHA * pa[i] + (1.0f - FUSION_ALPHA) * pb[i];
    }
}

static void fuse_f16_payload(
        std::vector<uint8_t> & fused,
        const std::vector<uint8_t> & a,
        const std::vector<uint8_t> & b,
        const payload_range & r) {
    if (r.nbytes % sizeof(ggml_fp16_t) != 0) {
        throw std::runtime_error("F16 payload byte count is not divisible by sizeof(ggml_fp16_t)");
    }

    const size_t n = r.nbytes / sizeof(ggml_fp16_t);
    const ggml_fp16_t * pa = checked_ptr_const<ggml_fp16_t>(a, r);
    const ggml_fp16_t * pb = checked_ptr_const<ggml_fp16_t>(b, r);
    ggml_fp16_t * pd = checked_ptr_mut<ggml_fp16_t>(fused, r);

    for (size_t i = 0; i < n; ++i) {
        const float va = ggml_fp16_to_fp32(pa[i]);
        const float vb = ggml_fp16_to_fp32(pb[i]);
        pd[i] = ggml_fp32_to_fp16(FUSION_ALPHA * va + (1.0f - FUSION_ALPHA) * vb);
    }
}

static std::vector<uint8_t> fuse_state_buffers(
        const std::vector<uint8_t> & state_a,
        const std::vector<uint8_t> & state_b,
        const state_layout & layout_a,
        const state_layout & layout_b) {
    if (state_a.size() != state_b.size()) {
        throw std::runtime_error("state buffers have different sizes; Level 1 PoC requires equal sizes");
    }

    require_same_layout(layout_a, layout_b);

    std::vector<uint8_t> fused_buffer(state_a.size());

    // Safety base: copy all magic/header/metadata bytes from state A. The fusion
    // loop below mutates only typed K/V payload ranges. llama_state_seq_set_data()
    // ignores the source seq_id in this header and installs the state into the
    // destination seq_id supplied by the caller.
    std::memcpy(fused_buffer.data(), state_a.data(), state_a.size());

    size_t fused_payload_count = 0;
    size_t fused_payload_bytes = 0;

    for (const payload_range & r : layout_a.payloads) {
        ensure_range(state_a, r);
        ensure_range(state_b, r);
        ensure_range(fused_buffer, r);

        if (r.type == (int32_t) GGML_TYPE_F32) {
            fuse_f32_payload(fused_buffer, state_a, state_b, r);
        } else if (r.type == (int32_t) GGML_TYPE_F16) {
            fuse_f16_payload(fused_buffer, state_a, state_b, r);
        } else {
            throw std::runtime_error("unsupported KV payload type; Level 1 PoC supports only GGML_TYPE_F16/F32");
        }

        fused_payload_count++;
        fused_payload_bytes += r.nbytes;
    }

    std::fprintf(stderr, "\nfused %zu K/V payload ranges, total payload bytes = %zu\n",
            fused_payload_count, fused_payload_bytes);

    return fused_buffer;
}

static bool inject_fused_state(llama_context * ctx, const std::vector<uint8_t> & fused_buffer) {
    const size_t n_set = llama_state_seq_set_data(ctx, fused_buffer.data(), fused_buffer.size(), SEQ_FUSED);
    if (n_set != fused_buffer.size()) {
        std::fprintf(stderr,
                "llama_state_seq_set_data failed or wrote partial data: got=%zu expected=%zu\n",
                n_set, fused_buffer.size());
        return false;
    }

    std::fprintf(stderr, "injected fused state into seq_id=%d, bytes=%zu\n", SEQ_FUSED, n_set);
    return true;
}

static bool verify_forward_decode(
        llama_context * ctx,
        const llama_vocab * vocab,
        llama_token seed_token,
        llama_pos start_pos,
        int32_t n_steps) {
    llama_sampler * smpl = llama_sampler_init_greedy();
    if (!smpl) {
        std::fprintf(stderr, "failed to create greedy sampler\n");
        return false;
    }

    llama_token token = seed_token;

    for (int32_t step = 0; step < n_steps; ++step) {
        const llama_pos pos = start_pos + step;
        if (!decode_one_token(ctx, token, pos, SEQ_FUSED)) {
            llama_sampler_free(smpl);
            return false;
        }

        const float * logits = llama_get_logits(ctx);
        if (!logits) {
            std::fprintf(stderr, "llama_get_logits returned null during verification\n");
            llama_sampler_free(smpl);
            return false;
        }

        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        int32_t argmax_id = 0;
        float argmax_logit = -std::numeric_limits<float>::infinity();
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (logits[i] > argmax_logit) {
                argmax_logit = logits[i];
                argmax_id = i;
            }
        }

        const llama_token sampled = llama_sampler_sample(smpl, ctx, -1);
        std::printf(
                "[verify] step=%d seq_id=%d input_token=%d pos=%d argmax_token=%d argmax_logit=%f sampled_token=%d piece='%s'\n",
                step,
                SEQ_FUSED,
                token,
                pos,
                argmax_id,
                argmax_logit,
                sampled,
                token_to_piece(vocab, sampled).c_str());
        std::fflush(stdout);

        if (llama_vocab_is_eog(vocab, sampled)) {
            std::fprintf(stderr, "verification reached EOG at step=%d\n", step);
            break;
        }

        token = sampled;
    }

    llama_sampler_free(smpl);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    app_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens_a;
    std::vector<llama_token> tokens_b;
    try {
        tokens_a = make_exact_length_tokens(vocab, params.prompt_a, params.n_tokens, "Prompt A");
        tokens_b = make_exact_length_tokens(vocab, params.prompt_b, params.n_tokens, "Prompt B");
    } catch (const std::exception & err) {
        std::fprintf(stderr, "tokenization failed: %s\n", err.what());
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_seq_max = 3;
    ctx_params.n_ctx     = std::max<uint32_t>(512, (uint32_t) params.n_tokens * ctx_params.n_seq_max + (uint32_t) params.n_verify + 32);
    ctx_params.n_batch   = std::max<uint32_t>(32, (uint32_t) params.n_tokens);
    ctx_params.n_ubatch  = ctx_params.n_batch;
    ctx_params.kv_unified = true;
    ctx_params.no_perf   = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::fprintf(stderr, "failed to create llama_context\n");
        llama_model_free(model);
        return 1;
    }

    std::fprintf(stderr,
            "context ready: n_ctx=%u, n_batch=%u, n_ubatch=%u, n_seq_max=%u\n",
            llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx), llama_n_seq_max(ctx));

    if (!decode_tokens_to_seq(ctx, tokens_a, SEQ_A)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::fprintf(stderr, "decoded Prompt A into seq_id=%d\n", SEQ_A);

    if (!decode_tokens_to_seq(ctx, tokens_b, SEQ_B)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::fprintf(stderr, "decoded Prompt B into seq_id=%d\n", SEQ_B);

    std::vector<uint8_t> state_a;
    std::vector<uint8_t> state_b;
    try {
        const size_t state_size_a = llama_state_seq_get_size(ctx, SEQ_A);
        const size_t state_size_b = llama_state_seq_get_size(ctx, SEQ_B);
        std::fprintf(stderr, "seq_id=%d state size = %zu bytes\n", SEQ_A, state_size_a);
        std::fprintf(stderr, "seq_id=%d state size = %zu bytes\n", SEQ_B, state_size_b);

        state_a = export_seq_state_host(ctx, SEQ_A);
        state_b = export_seq_state_host(ctx, SEQ_B);
    } catch (const std::exception & err) {
        std::fprintf(stderr, "state export failed: %s\n", err.what());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::vector<uint8_t> fused_buffer;
    try {
        const state_layout layout_a = parse_state_layout(state_a, "seq0");
        const state_layout layout_b = parse_state_layout(state_b, "seq1");
        fused_buffer = fuse_state_buffers(state_a, state_b, layout_a, layout_b);
    } catch (const std::exception & err) {
        std::fprintf(stderr, "state fusion failed: %s\n", err.what());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (!inject_fused_state(ctx, fused_buffer)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_token seed_token = tokens_a.back();
    std::fprintf(stderr,
            "\nstarting fused-memory forward verification: seed_token=%d piece='%s', start_pos=%d, steps=%d\n",
            seed_token,
            token_to_piece(vocab, seed_token).c_str(),
            params.n_tokens,
            params.n_verify);

    if (!verify_forward_decode(ctx, vocab, seed_token, params.n_tokens, params.n_verify)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::fprintf(stderr, "\nCacheBlend static fusion PoC completed without crash.\n");

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
