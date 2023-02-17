#include "dlib/image_processing/full_object_detection.h"
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

void FaceRecognition::_raw_face_locations(
	dlib::matrix<dlib::rgb_pixel> &img, std::pair<int, int> res,
	std::vector<dlib::mmod_rect> &face_locations, std::string model)
{
	// Upsampling the image will allow us to detect smaller faces but will
	// cause the program to use more RAM and run longer.
	if (res.first != -1 && res.second != -1)
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

		face_locations = cnn_face_detector(img);
	}
	else
	{
		// TODO:  <14-02-23, Arjun>
		throw std::runtime_error("Unimplemented HOG model");
	}
}
void FaceRecognition::_batched_raw_face_locations(
	std::vector<dlib::matrix<dlib::rgb_pixel>> &imgs, std::pair<int, int> res,
	std::vector<std::vector<dlib::mmod_rect>> &face_locations,
	std::string model)
{
	// Upsampling the image will allow us to detect smaller faces but will
	// cause the program to use more RAM and run longer.
	if (res.first != -1 && res.second != -1)
		for (auto &img : imgs)
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

		face_locations = cnn_face_detector(imgs);
	}
	else
	{
		// TODO:  <14-02-23, Arjun>
		throw std::runtime_error("Unimplemented HOG model");
	}
}
// TODO: fix this function, it is currentl wrong
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
void FaceRecognition::scan_known_people(
	const std::filesystem::path &known_folder, const std::pair<int, int> &res)
{
	std::vector<std::pair<std::string, std::string>> image_files;
	_get_image_files_in_directory(known_folder, image_files);
	std::vector<matrix<rgb_pixel>> faces;

	for (auto &file : image_files)
	{
		matrix<rgb_pixel> img;
		dlib::load_image<dlib::matrix<dlib::rgb_pixel>>(img, file.first);
		faces.push_back(img);
	}
	std::vector<std::vector<dlib::mmod_rect>> detected_faces;
	_batched_raw_face_locations(faces, res, detected_faces, "cnn");
	std::vector<matrix<rgb_pixel>> shape_normalized_faces;
	for (size_t i = 0; i < detected_faces.size(); i++)
	{
		auto detected_face = detected_faces[i];
		if (detected_face.size() == 0)
		{
			printf("No faces found.\n");
			continue;
		}
		else if (detected_face.size() > 1)
		{
			printf(
				"More than one face found. Considering only the first face.\n");
			continue;
		}
		// Refer: http://dlib.net/dnn_face_recognition_ex.cpp.html
		dlib::matrix<dlib::rgb_pixel> face_img = faces[i];
		auto shape = pose_predictor_5_point(face_img, detected_face[0]);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(face_img, get_face_chip_details(shape, 150, 0.25),
						   face_chip);
		shape_normalized_faces.push_back(std::move(face_chip));
	}
	std::vector<matrix<float, 0, 1>> face_descriptors =
		face_encoder(shape_normalized_faces);
}
void FaceRecognition::_get_image_files_in_directory(
	const std::filesystem::path &known_folder,
	std::vector<std::pair<std::string, std::string>> &image_files)
{
	namespace fs = std::filesystem;
	const std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png"};
	for (const auto &file : fs::directory_iterator(known_folder))
	{
		if (fs::is_regular_file(file.path()))
		{
			std::string file_extension = file.path().extension().string();
			std::string basename = file.path().filename();
			if (std::find(image_extensions.begin(), image_extensions.end(),
						  file_extension) == image_extensions.end())
			{
				std::cout << file.path().string() << std::endl;
				image_files.push_back({file.path().string(), basename});
			}
		}
	}
}
