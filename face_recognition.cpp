#include "face_recognition.hpp"
std::tuple<long, long, long, long>
FaceRecognition::_trim_css_to_bounds(std::tuple<long, long, long, long> css,
									 dlib::matrix<dlib::rgb_pixel> &img)

{
	long width = img.nc();
	long height = img.nr();
	return std::make_tuple(
		std::max(std::get<0>(css), 0L), std::min(std::get<1>(css), width),
		std::min(std::get<2>(css), height), std::max(std::get<3>(css), 0L));
}

std::vector<dlib::mmod_rect>
FaceRecognition::_raw_face_locations(dlib::matrix<dlib::rgb_pixel> &img,
									 std::pair<int, int> res, std::string model)
{
	std::vector<dlib::mmod_rect> face_locations;
	// Upsampling the image will allow us to detect smaller faces but will
	// cause the program to use more RAM and run longer.

	while (img.size() < res.first * res.second)
		pyramid_up(img);

	if (model == "cnn")
	{
		// Note that you can process a bunch of images in a std::vector at once
		// and it runs much faster, since this will form mini-batches of images
		// and therefore get better parallelism out of your GPU hardware.
		// However, all the images must be the same size.  To avoid this
		// requirement on images being the same size we process them
		// individually in this example.

		auto temp = cnn_face_detector(img);
		for (auto &&d : temp)
		{
			face_locations.push_back(d);
		}
	}
	else
	{
		// TODO:  <14-02-23, Arjun>
		throw std::runtime_error("Unimplemented HOG model");
	}
	return face_locations;
}
