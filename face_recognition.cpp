#include "face_recognition.hpp"
#include <filesystem>
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
void FaceRecognition::recognize_faces(std::vector<dlib::mmod_rect> &faces)
{
	std::vector<matrix<float, 0, 1>> face_descriptors = face_encoder(faces);
	if (faces.size() == 0)
	{
		printf("No faces found.\n");
		return;
	}
	std::vector<sample_pair> edges;
	for (size_t i = 0; i < face_descriptors.size(); ++i)
	{
		for (size_t j = i; j < face_descriptors.size(); ++j)
		{
			// Faces are connected in the graph if they are close enough.  Here
			// we check if the distance between two face descriptors is less
			// than 0.6, which is the decision threshold the network was trained
			// to use.  Although you can certainly use any other threshold you
			// find useful.
			if (length(face_descriptors[i] - face_descriptors[j]) < 0.6)
				edges.push_back(sample_pair(i, j));
		}
	}
	std::vector<unsigned long> labels;
	const auto num_clusters = chinese_whispers(edges, labels);
	// This will correctly indicate that there are 4 people in the image.
	std::cout << "number of people found in the image: " << num_clusters
			  << std::endl;
}
void FaceRecognition::_get_image_files_in_directory(
	std::filesystem::path &known_folder, std::vector<std::string> &image_files)
{
	namespace fs = std::filesystem;
	const std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png"};
	for (const auto &file : fs::directory_iterator(known_folder))
	{
		if (fs::is_regular_file(file.path()))
		{
			std::string file_extension = file.path().extension().string();
			if (std::find(image_extensions.begin(), image_extensions.end(),
						  file_extension) == image_extensions.end())
			{
				std::cout << file.path().string() << std::endl;
				image_files.push_back(file.path().string());
			}
		}
	}
}
