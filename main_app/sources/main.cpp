#include "tests.h"
int main()
{
    TestSimpleModel();
    TestOneEncoder();
    TestSimpleEncoderDecoder();
    TestSimpleEncoderDecoderLSTM();
    TestDNS48();
    TestDemucsFullForwardAudio();
    TestStreamerForwardAudio();
    TestResampler();
    GenerateTestsforAudio();
    return 0;
}