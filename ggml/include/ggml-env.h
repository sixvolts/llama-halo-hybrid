#pragma once

// halo-hybrid (fork-local, not upstream API): one parse convention for the fork's environment switches.
//
//   ggml_env_flag(name, def)  unset: def; "", "0", "false", "off", "no" (any case): false; any other value: true
//   ggml_env_int(name, def)   unset: def; a whole-string integer: its value; anything else: def, with one warning
//   ggml_env_float(name, def) as ggml_env_int, for a floating-point value
//
// The environment is read when called; callers cache the result in a function-local static.

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

GGML_API bool    ggml_env_flag (const char * name, bool    def);
GGML_API int64_t ggml_env_int  (const char * name, int64_t def);
GGML_API double  ggml_env_float(const char * name, double  def);

// every GGML_* and LLAMA_* variable set in the environment, as " NAME=value ..." (empty when none): LLAMA_ARG_* is left
// out (it mirrors command-line options, --api-key among them) and names with KEY/TOKEN/SECRET/PASS show <redacted>.
// Computed once; a log line with it lets a bug report carry the switches a run used (typos included)
GGML_API const char * ggml_env_set_string(void);

// log ggml_env_set_string() at INFO, once per process (llama-server prints it itself: its default verbosity hides
// library INFO lines)
GGML_API void    ggml_env_log_set(const char * who);

#ifdef __cplusplus
}
#endif
