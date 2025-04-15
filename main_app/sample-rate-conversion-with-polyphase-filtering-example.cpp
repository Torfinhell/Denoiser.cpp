#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <sndfile.h>

#ifdef WITH_RLIMIT
#include <sys/resource.h>
#endif


//// Common types ////

typedef float sample_type;
typedef std::vector < sample_type > samples;


//// Math functions ////

static float const pi = 3.1415926535f;

unsigned int gcd(unsigned int p_first, unsigned int p_second)
{
	if (p_first == 0) return p_second;
	if (p_second == 0) return p_first;

	do
	{
		unsigned int temp = p_first % p_second;
		p_first = p_second;
		p_second = temp;
	}
	while (p_second != 0);

	return p_first;
}

unsigned int lcm(unsigned int p_first, unsigned int p_second)
{
	unsigned int temp = gcd(p_first, p_second);
	return (temp != 0) ? (p_first * p_second / temp) : 0;
}

float sinc(float f)
{
	return (f == 0.0f) ? 1.0f : (std::sin(f * pi) / (f * pi));
}

float hanning_func(int n, int N)
{
	return 0.5f * (1.0f - std::cos(2.0f * pi * float(n) / float(N - 1)));
}

samples compute_hanning_window(std::size_t const p_window_size)
{
	samples window(p_window_size, 0);
	for (std::size_t i = 0; i < p_window_size; ++i)
		window[i] = hanning_func(i, p_window_size);
	return window;
}

sample_type convolve(sample_type const *p_first_samples, sample_type const *p_second_samples, std::size_t const p_num_samples)
{
	sample_type result = 0.0f;
	int num_samples = int(p_num_samples);

	for (int first_index = 0; first_index < num_samples; ++first_index)
	{
		int second_index = num_samples - 1 - first_index;

		sample_type first_sample = p_first_samples[first_index];
		sample_type second_sample = p_second_samples[second_index];
		result += first_sample * second_sample;
	}

	return result;
}

samples compute_polyphase_filter_bank(std::size_t const p_upsampling_factor, std::size_t const p_downsampling_factor, std::size_t const p_filter_size)
{
	// Create the window function that will be applied on top of the
	// stretched sinc. The window size equals the filter size plus the
	// upsampling factor, since we need to account for the extra nullsamples that
	// get inserted during zerostuffing.
	samples window = compute_hanning_window(p_filter_size * p_upsampling_factor);

	samples filter(window.size(), 0);

	// We need to create a low pass filter that cuts off frequencies at half of
	// either the input or the output sample rate, whichever is lower. However,
	// it turns out that for the filter calculation, we don't really need the
	// sample rates directly. Instead, we only need to "stretch" sinc by a
	// specific factor. A factor of 1 means no change. A factor of 2 means that
	// half of the original frequency range is cut off etc.
	//
	// If we only want to upsample by a factor of 2, and don't want to downsample,
	// then it is sufficient to stretch the sinc by a factor of 2. This is because
	// when we upsample, we apply zero stuffing, which creates unwantd spectral
	// images above the desired range. So, if for example we upsample by a factor
	// of 2, this means we insert 2-1 = 1 nullsample in between the original
	// samples (this is the zero-stuffing part). Like this:
	// 
	// a b c d e f ... -> a 0 b 0 c 0 d 0 e 0 f 0 ...
	//
	// The lower 50% of the spectrum contains the original spectral image. So, we
	// need to get rid of anything except the lower half in the zero-stuffed signal.
	//
	// Another example: Upsampling by a factor of 3, no downsampling. Here,
	// we have 2 unwanted spectral images above the original one, and 3-1 = 2
	// nullsamples were stuffed in between the original samples. Like this:
	//
	// a b c d ... -> a 0 0 b 0 0 c 0 0 d 0 0 ...
	//
	// Since we only want the original image, and its spectrum only makes 1/3rd
	// of the spectrum of the zerostuffed signal, we stretch sinc by a factor of
	// 3, which causes it to lowpass-filter anything except the lowest 33,3% of
	// the spectrum.
	//
	// If we also downsample, then it depends by how much. If the downsampling
	// factor is higher than the upsampling, it means that the downsampled
	// signal will have a Nyquist frequency that is *lower* than that of the
	// original signal. Example: converting from 24000 Hz to 22050 Hz. Here,
	// we need to lowpass-filter to make sure we get rid of all frequencies
	// above 22050/2 = 11025 Hz, otherwise aliasing will occur in the downsampled
	// signal.
	//
	// If instead we downsample to a rate that is *higher* than the original
	// one, we lowpass-filter with the original signal's Nyquist frequency.
	// If we convert from 24000 Hz to 44100 Hz, we lowpass-filter to get rid
	// of all frequencies above 24000 / 2 = 12000 Hz.
	//
	// Again, the actual sample rate does not matter, only the factors do.
	// We simply pick the higher of the two (up/downsampling factors). This
	// corresponds to picking the lower of the two sample rates (input/output
	// sample rates).
	float scale_factor = float(std::max(p_downsampling_factor, p_upsampling_factor));

	// The polyphase filter bank is stored as a one-dimensional array.
	// The polyphase filters are arranged in the array as shown in
	// this example:
	//
	//   AAABBBCCCDDD
	//
	// Where A,B,C,D are the coefficients of polyphase filters A,B,C,D.
	// Upsampling factor is 4 (that's why there are four filters). Each
	// filter has 3 taps in this example.

	// NOTE: It would be useful to reverse the filters, that is,
	// coefficients 1234 -> 4321, to make convolution easier, because
	// then it can be done using a simple multiply-and-add operation.
	// (Convolution does multiply-and-add, but with one of the inputs
	// reversed, so by pre-reversing one, it devolves into a simple
	// multiply-and-add.)

	std::size_t num_filters = p_upsampling_factor;
	std::size_t num_filter_taps = window.size() / num_filters;

	for (std::size_t i = 0; i < window.size(); ++i)
	{
		// Note that we do _not_ scale t by the window size here. So, we do not
		// normalize it in any way. This is intentional: The filter size defines
		// the _quality_ of the filtering. Larger filter means more sinc coefficients,
		// or in other words, we tap a larger range of sinc. If we normalized t,
		// we would always tap the _same_ range of the sinc function (the area
		// around the main lobe).
		// We do however offset t to make sure it ranges from -w/2 to w/2-1 instead of
		// 0 to w-1 . Otherwise we don't get a symmetric sinc tap.
		// TODO: Should it be from -w to +w instead?
		float t = int(i) - int(window.size() / 2);

		// Compute stretched sinc.
		float f = sinc(t / scale_factor);

		std::size_t polyphase_filter_index = i % num_filters;
		std::size_t offset_in_polyphase_filter = i / num_filters;
		std::size_t filter_array_idx = polyphase_filter_index * num_filter_taps + offset_in_polyphase_filter;

		// Apply the window function on top of the sinc
		// function to produce filter coefficients.
		filter[filter_array_idx] = f * window[i];
	}

	// TODO: It is unclear why this is necessary, but without this,
	// the output signal may be incorrectly amplified.
	if (p_downsampling_factor > p_upsampling_factor)
	{
		float dampening_factor = float(p_upsampling_factor) / p_downsampling_factor;
		for (auto & filter_coefficient : filter)
			filter_coefficient *= dampening_factor;
	}

	return filter;
}



//// main ////

int main(int argc, char *argv[])
{
	// Get the command line arguments.
	if (argc < 5)
	{
		std::cerr << "Usage: " << argv[0] << " <input filename> <output WAV filename> <output sample rate> <filter size>\n";
		return -1;
	}

	std::string input_filename = argv[1];
	std::string output_filename = argv[2];
	unsigned int output_sample_rate = std::stoul(argv[3]);
	std::size_t filter_size = std::stoul(argv[4]);


#ifdef WITH_RLIMIT
	// Limit the amount of memory this process can allocate to 1 GB to
	// prevent lots of swap activity in case we allocate too much. Since
	// this example just loads the entirety of the input file to memory,
	// this can happen with long tracks. With this rlimit, the process
	// is killed as soon as the limit is hit.
	struct rlimit memlim;
	getrlimit(RLIMIT_AS, &memlim);
	memlim.rlim_cur = std::min(std::size_t(1024*1024*1024), memlim.rlim_cur);
	setrlimit(RLIMIT_AS, &memlim);
#endif


	// Open the input file.
	// NOTE: In this example, we read and sample rate convert the entire
	// input file at once. This is for sake of clarity and simplicity.
	// In production, the conversion would not be done that way. Instead,
	// the input samples would be streamed in and converted on the fly.
	// This saves a ton of memory, and would also work with live streams.

	sound_file input_sound_file;
	if (!input_sound_file.open_for_reading(input_filename))
		return -1;
	samples input_samples;
	input_sound_file.read_all_samples(input_samples);
	if (input_samples.empty())
	{
		std::cerr << "Did not get any input samples\n";
		return -1;
	}

	unsigned int input_sample_rate = input_sound_file.get_sample_rate();


	// Sample rate conversion works by first interpolating the signal
	// to a higher sample rate, then decimating to a lower sample rate.
	//
	// Interpolation means that new samples are introduced in between
	// the original input samples, followed by a lowpass filter that is
	// applied on that augmented input signal. The signal is augmented
	// by inserting zeros in between the original samples. This is
	// called "zero-stuffing". Zero-stuffing introduces copies of the
	// original spectrum, _above_ the original spectrum. This is where
	// the low-pass filter comes in - it cuts off these unwanted copies,
	// leaving us with an upsampled version of the original signal.
	//
	// So, if we want to upsample the signal by an integer factor N,
	// we insert N-1 nullsamples between the original samples.
	//
	// Decimation can then be applied. This simply involves picking
	// every Mth sample, M being the decimation factor. M=1 means no
	// decimation.
	//
	// In short, first we upsample by N by applying interpolation,
	// then we downsample by M by applying decimation. The M/N ratio
	// is the overall sample rate conversion ratio. Upsampling factor
	// N is also the interpolation factor N. Downsampling factor M
	// is also the decimation factor M.
	//
	// The up/downsampling factors are derived from the input and
	// output sample rates. To that end, the least common denominator
	// of the sample rates is determined. That's because the up- and
	// downsampling factors influence filter sizes, and we want filters
	// to not be unnecessarily large. For example, input sample rate
	// 48000 Hz, output sample rate 44100 Hz, that's a ratio of
	// 48000/44100. We could use N=48000, M=44100, but for better
	// efficiency (as explained above), we reduce this and come up
	// with an equal ratio of 160/147.
	//
	// Once the factors are known, we could in theory get the input
	// signal's samples and stuff in nullsamples between these samples.
	// However, that would be wasteful (in the example above it would
	// increase the input signal size by a factor of 160). We can
	// make the observation that during convolution, these nullsamples
	// don't contribute to the output, because these nullsamples are
	// multiplied with filter coefficients and added. So, an equation
	// like:
	//
	//   output = sample * coeff1 + 0 * coeff2 + 0 * coeff3 ...
	//
	// will only really be influenced by the original samples, not
	// by the added nullsamples.
	//
	// The idea then is to simply omit these nullsamples and only
	// pick the coefficients that would actually be applied (in the
	// example above, coeff1 would be applied, while coeff2 and
	// coeff3 would not). We observe that in a zerostuffed signal,
	// the original samples appear in every Nth position. So, in this
	// zerostuffed signal:
	//
	// a 0 0 b 0 0 c ..
	//
	// every 3rd sample is an original sample. If we have a filter
	// with 12 taps, then its coefficients can be decomposed into
	// subfilters. The number of filters equal the upsampling factor.
	// In this example, the decomposition would be:
	//
	// Original filter: h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12
	// Subfilter 1:     h1       h4       h7       h10
	// Subfilter 2:        h2       h5       h8        h11
	// Subfilter 3:           h3       h6       h9         h12
	//
	// During decimation, we would normally pick a sample from the
	// interpolated signal. Suppose that we didn't use polyphase
	// filters, but instead actually did use zerostuffing and
	// filtered that zerostuffed signal. Suppose upsampling factor
	// N is 3, downsampling factor M is 2.
	//
	// Original signal: i1 i2 i3 i4 i5 i6 i7 ..
	// Zerostuffed signal: i1 0 0 i2 0 0 i3 0 0 i4 0 0 i5 0 0 ..
	// Interpolated signal: i1 a b i2 c d i3 e f i4 g h i6 i j ..
	//   (a-j are newly interpolated samples that resulted from
	//   convolving the filter with the zerostuffed signal)
	//
	// Decimation factor M = 2 means we pick every 2nd
	// interpolated sample, and get the output signal
	// i1 b c i3 f g i6 ..
	//
	// We optimize this by using polyphase filters, eliminating
	// the need for an intermediate zerostuffed signal. Instead,
	// we pick the subfilter that would produce the sample that
	// we pick during decimation.
	//
	// From the example above, suppose we are about to pick
	// sample "b". to compute "b", we pick the appropriate
	// subfilter. We find this subfilter by computing the
	// position in the interpolated signal. This we call the
	// "interpolated position". In the case of "b", this position
	// would be 2. (Positions start at 0.) We then apply modulo 3
	// (the number of subfilters) to pick the appropriate subfilter.
	// That's subfilter (2 mod 3) = 2 in this case.
	//
	// Then we need the position in the original input signal that
	// corresponds to the "b" sample". This we get by calculating
	// position_in_original_signal = position_in_output_signal * M / N
	// So, in this case, "b" is at position #1 in the output signal,
	// and 1*2/3 = 0. Now we have the position in the input signal
	// where we get the input samples that we want to convolve with
	// the subfilter #2 we picked earlier. This means that this
	// subfilter is convolved with input samples i1 to i4.
	//
	// Another example would be if we wanted to compute sample "g",
	// which is at "interpolated position" position 10 and at output
	// position 5. 10 mod 3 = 1, and 5*2/3 = 3. So, we would convolve
	// input samples i4 to i7 with subfilter #1.


	// Compute the up- and downsampling factors.
	unsigned int sample_rate_lcm = lcm(input_sample_rate, output_sample_rate);
	// This is the interpolation factor N mentioned above.
	unsigned int upsampling_factor = sample_rate_lcm / input_sample_rate;
	// This is the decimation factor M mentioned above.
	unsigned int downsampling_factor = sample_rate_lcm / output_sample_rate;


	// Print some info.
	std::cerr << "Input / output sample rate: " << input_sample_rate << " Hz / " << output_sample_rate << " Hz\n";
	std::cerr << "Up / downsampling factors: " << upsampling_factor << " / " << downsampling_factor << "\n";
	std::cerr << "Filter size: " << filter_size << "\n";


	// Prepare the polyphase filter and the output samples buffer.
	samples polyphase_filter_bank = compute_polyphase_filter_bank(upsampling_factor, downsampling_factor, filter_size);
	samples output_samples(input_samples.size() * upsampling_factor / downsampling_factor, 0);

	std::size_t num_polyphase_filters = upsampling_factor;
	std::size_t polyphase_filter_size = polyphase_filter_bank.size() / num_polyphase_filters;


	// Open the output file.
	sound_file output_sound_file;
	if (!output_sound_file.open_for_writing(output_filename, output_sample_rate))
		return -1;


	// Now perform the sample rate conversion.
	std::size_t output_position = 0;
	std::size_t interpolated_position = 0;
	std::size_t last_progress = 0;
	// Don't iterate over the last filter_size samples. That's because
	// convolve() will access all samples from output_position to
	// (output_position + polyphase_filter_size - 1).
	for (; output_position < output_samples.size() - filter_size; ++output_position, interpolated_position += downsampling_factor)
	{
		// Print some progress. We keep track of the last computed
		// progress to check if we should actually print something.
		// Otherwise, the constant console output could actually
		// slow down this code (and it floods the console with
		// lines of course).
		std::size_t progress = output_position / 100000;
		if (progress != last_progress)
		{
			std::cerr << output_position << "/" << output_samples.size() << "(" << (float(output_position) / float(output_samples.size()) * 100.0f) << "%)\n";
			last_progress = progress;
		}

		// Pick the right subfilter.
		std::size_t polyphase_filter_index = interpolated_position % num_polyphase_filters;
		sample_type const *polyphase_filter_coefficients = &(polyphase_filter_bank[polyphase_filter_index * polyphase_filter_size]);

		// Pick what input samples we need to convolve with the
		// chosen subfilter.
		sample_type const *input_samples_coefficients = &(input_samples[output_position * downsampling_factor / upsampling_factor]);

		// Perform the convolution, producing the sample rate
		// converted output.
		output_samples[output_position] = convolve(input_samples_coefficients, polyphase_filter_coefficients, polyphase_filter_size);
	}


	// Write the result.
	output_sound_file.write_samples(output_samples, output_samples.size());


	return 0;
}
