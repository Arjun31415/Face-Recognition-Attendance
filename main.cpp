// #include "face_recognition.hpp"
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

#include "models/models.hpp"
#include <iostream>
using namespace dlib;
template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<
	con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<
	con<1, 9, 9, 1, 1,
		rcon5<rcon5<
			rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

int main(int argc, char **argv)
try
{
	if (argc == 1)
	{
		std::cout << "Call this program like this:"
				  << "\n";
		std::cout
			<< "./dnn_mmod_face_detection_ex mmod_human_face_detector.dat "
			   "faces/*.jpg"
			<< "\n";
		std::cout
			<< "\nYou can get the mmod_human_face_detector.dat file from:\n";
		std::cout << "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
				  << "\n";
		return 0;
	}

	net_type net;
	deserialize(argv[1]) >> net;

	image_window win;
	for (int i = 2; i < argc; ++i)
	{
		matrix<rgb_pixel> img;
		load_image(img, argv[i]);

		// Upsampling the image will allow us to detect smaller faces but will
		// cause the program to use more RAM and run longer.
		/* while (img.size() < 1800 * 1800)
			pyramid_up(img); */

		// Note that you can process a bunch of images in a std::vector at once
		// and it runs much faster, since this will form mini-batches of images
		// and therefore get better parallelism out of your GPU hardware.
		// However, all the images must be the same size.  To avoid this
		// requirement on images being the same size we process them
		// individually in this example.
		auto dets = net(img);
		win.clear_overlay();
		win.set_image(img);
		for (auto &&d : dets)
			win.add_overlay(d);

		std::cout << "Hit enter to process the next image."
				  << "\n";
		std::cin.get();
	}
}
catch (std::exception &e)
{
	std::cout << e.what() << "\n";
}
