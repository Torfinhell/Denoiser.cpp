#include "layers.h"
#include "tests.h"
int main()
{
    TestSimpleModel();
    TestOneEncoder();
    TestSimpleEncoderDecoder();
    TestSimpleEncoderDecoderLSTM();
    TestBasicDemucsModel();
    // TestDemucsForwardAudio();
    // TestStreamerForwardAudio();
    // TestResampler();
    // GenerateTestsforAudio();
    return 0;
}