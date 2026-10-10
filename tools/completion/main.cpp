#include "log.h"

int llama_completion(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int ret = llama_completion(argc, argv);
    // the logger is never destroyed (common_log_main): drain it on every return, or under CPU load the process exits
    // with the last generated tokens (or the error that ended the run) still queued and stdout/stderr come up short
    common_log_pause(common_log_main());
    return ret;
}
