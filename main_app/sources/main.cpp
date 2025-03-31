#include "audio.h"
#include "layers.h"
#include "tensors.h"
#include "tests.h"
#include <mutex>
#include <vector>
int main()
{
    // TestSimpleModel();
    // TestOneEncoder();
    // TestSimpleEncoderDecoder();//doesnt work
    // TestSimpleEncoderDecoderLSTM();
    // TestBasicDemucsModel();
    TestDemucsStreamer();
    return 0;
}