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
	std::cout << "Image size: " << imgs.size() << std::endl;
	if (model == "cnn")
	{
		// Note that you can process a bunch of images in a std::vector at once
		// and it runs much faster, since this will form mini-batches of images
		// and therefore get better parallelism out of your GPU hardware.
		// However, all the images must be the same size.  To avoid this
		// requirement on images being the same size we process them
		// individually in this example.

		face_locations = cnn_face_detector(imgs);
		std::cout << face_locations.size() << std::endl;
	}
	else
	{
		// TODO:  <14-02-23, Arjun>
		throw std::runtime_error("Unimplemented HOG model");
	}
}
void FaceRecognition::recognize_faces(matrix<rgb_pixel> &img,
									  std::vector<dlib::mmod_rect> &faces,
									  std::vector<dlib::mmod_rect> &overlay,
									  std::vector<std::string> &names)
{
	if (faces.size() == 0)
	{
		printf("No faces found.\n");
		return;
	}
	while (img.size() < 1800 * 1800)
		pyramid_up(img);
	std::vector<sample_pair> edges;
	std::cout << "Faces to recognize: " << faces.size() << std::endl;
	std::vector<matrix<rgb_pixel>> shape_normalized_faces;
	for (size_t i = 0; i < faces.size(); i++)
	{
		auto detected_face = faces[i];
		/* if (detected_face.size() == 0)
		{
			printf("No faces found.\n");
			continue;
		}
		else if (detected_face.size() > 1)
		{
			printf(
				"More than one face found. Considering only the first face.\n");
			continue;
		} */
		// Refer: http://dlib.net/dnn_face_recognition_ex.cpp.html
		dlib::matrix<dlib::rgb_pixel> face_img = img;
		auto shape = pose_predictor_5_point(face_img, detected_face);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(face_img, get_face_chip_details(shape, 150, 0.25),
						   face_chip);
		shape_normalized_faces.push_back(std::move(face_chip));
		assert(shape_normalized_faces.size() == image_files.size());
	}
	std::vector<matrix<float, 0, 1>> face_descriptors =
		face_encoder(shape_normalized_faces);
	std::vector<matrix<float, 0, 1>> unknown_face_descriptors =
		face_encoder(shape_normalized_faces);

	for (size_t i = 0; i < unknown_face_descriptors.size(); ++i)
	{
		size_t recognised_person_idx = this->known_face_descriptors.size();
		float min_len = std::numeric_limits<float>::max();
		for (size_t j = 0; j < this->known_face_descriptors.size(); j++)
		{
			auto temp =
				length(unknown_face_descriptors[i] - known_face_descriptors[j]);
			std::cout << temp << "\n";

			if (temp < 0.6 && temp < min_len)
			{
				min_len = temp;
				recognised_person_idx = j;
			}
		}
		if (recognised_person_idx < this->known_face_descriptors.size())
		{
			overlay.push_back(faces[i]);
			names.push_back(this->known_face_names[recognised_person_idx]);
			std::cout << "Person Recognised ";
			std::cout << recognised_person_idx << " ";
			std::cout << "Must be "
					  << this->known_face_names[recognised_person_idx]
					  << std::endl;
		}
		else std::cout << "Unkown person\n";
	}
}
void FaceRecognition::batched_recognize_faces(
	vector<matrix<rgb_pixel>> &imgs, std::vector<dlib::mmod_rect> &faces,
	std::vector<dlib::mmod_rect> &overlay, std::vector<std::string> &names)
{
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
	std::cout << detected_faces.size() << std::endl;
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
		assert(shape_normalized_faces.size() == image_files.size());
	}
	for (size_t i = 0; i < shape_normalized_faces.size(); i++)
	{
		printf("Dude: %s\n", image_files[i].second.c_str());
		std::cout << shape_normalized_faces[i].size() << "\n";
	}
	this->known_face_descriptors = face_encoder(shape_normalized_faces);
	for (auto it = std::make_move_iterator(image_files.begin()),
			  end = std::make_move_iterator(image_files.end());
		 it != end; ++it)
	{
		this->known_face_names.push_back(std::move(it->second));
	}
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
			// std::cout << basename << " " << file_extension << "\n";
			if (std::find(image_extensions.begin(), image_extensions.end(),
						  file_extension) != image_extensions.end())
			{
				std::cout << file.path().string() << std::endl;
				image_files.push_back({file.path().string(), basename});
			}
		}
	}
}
