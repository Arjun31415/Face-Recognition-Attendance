/**
 * @file main.cpp
 * @brief the main driver code for smart attendance system
 * @author Arjun31415
 */

#include "dlib/gui_widgets/widgets.h"
#include "face_recognition.hpp"
#include "models/models.hpp"
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <iostream>
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
std::pair<int, int> res = {1000, 1000};
int main(int argc, char **argv)
try
{
	if (argc < 3)
	{
		std::cout << "Call this program like this:"
				  << "\n";
		std::cout << "./face_attendance"
					 "faces/*.jpg"
				  << "\n";
		std::cout
			<< "\nYou can get the mmod_human_face_detector.dat file from:\n";
		std::cout << "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
				  << "\n";
		return 0;
	}

	FaceRecognition recognizer = FaceRecognition();
	std::cout << "Scanning\n";
	recognizer.scan_known_people(std::filesystem::path(argv[1]), res);
	dlib::matrix<dlib::rgb_pixel> img;
	load_image(img, argv[2]);
	bool interactive = true;
	if (argc == 4)
	{

		strncmp(argv[3], "--nointeractive", 13) == 0 ? interactive = false
													 : interactive = true;
	}
	std::vector<mmod_rect> dets;
	std::cout << "Starting face_locations\n";
	recognizer._raw_face_locations(img, res, dets);
	std::vector<mmod_rect> overlays;
	std::vector<std::string> names;
	std::cout << "Starting recgonition\n";
	recognizer.recognize_faces(img, dets, overlays, names);
    std::cout << "Starting visual\n";
	image_window win;
	win.clear_overlay();
	win.set_image(img);
	for (size_t i = 0; i < overlays.size(); ++i)
		win.add_overlay(dlib::image_window::overlay_rect(
			overlays[i], rgb_pixel(255, 0, 0), names[i]));
	if (interactive)
	{
		std::cout << "Hit enter\n";
		std::cin.get();
	}
}
catch (std::exception &e)
{
	std::cout << e.what() << "\n";
}
