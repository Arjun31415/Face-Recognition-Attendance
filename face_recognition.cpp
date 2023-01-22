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
std::vector<dlib::rectangle>
FaceRecognition::_raw_face_locations(dlib::matrix<dlib::rgb_pixel> &img,
									 int number_of_times_to_upsample,
									 std::string model)
{
	std::vector<dlib::rectangle> face_locations;

	if (model == "cnn")
	{
		return cnn_face_detector(img, number_of_times_to_upsample);
	}
	else
	{
		return face_detector(img, number_of_times_to_upsample);
	}
}
std::vector<std::unordered_map<std::string, std::vector<dlib::point>>>
face_landmarks(dlib::matrix<dlib::rgb_pixel> &img,
			   std::vector<dlib::rectangle> face_locations =
				   std::vector<dlib::rectangle>(),
			   std::string model = "large")
{
	dlib::shape_predictor sp;
	if (model == "large")
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	else if (model == "small")
		dlib::deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	else
		throw std::runtime_error("Invalid landmarks model type. Supported "
								 "models are ['small', 'large'].");
	std::vector<std::unordered_map<std::string, std::vector<dlib::point>>>
		landmarks;
	if (face_locations.empty())
	{
		dlib::frontal_face_detector detector =
			dlib::get_frontal_face_detector();
		face_locations = detector(img);
	}
	for (auto &loc : face_locations)
	{
		dlib::full_object_detection shape = sp(img, loc);
		std::unordered_map<std::string, std::vector<dlib::point>>
			face_landmarks;
		if (model == "large")
		{
			face_landmarks["chin"] =
				std::vector<dlib::point>(shape.part(0), shape.part(17));
			face_landmarks["left_eyebrow"] =
				std::vector<dlib::point>(shape.part(17), shape.part(22));
			face_landmarks["right_eyebrow"] =
				std::vector<dlib::point>(shape.part(22), shape.part(27));
			face_landmarks["nose_bridge"] =
				std::vector<dlib::point>(shape.part(27), shape.part(31));
			face_landmarks["nose_tip"] =
				std::vector<dlib::point>(shape.part(31), shape.part(36));
			face_landmarks["left_eye"] =
				std::vector<dlib::point>(shape.part(36), shape.part(42));
			face_landmarks["right_eye"] =
				std::vector<dlib::point>(shape.part(42), shape.part(48));
			std::vector<dlib::point> top_lip, bottom_lip;
			for (auto i = 48; i < 55; i++)
				top_lip.push_back(shape.part(i));
			top_lip.push_back(shape.part(64));
			top_lip.push_back(shape.part(63));
			top_lip.push_back(shape.part(62));
			top_lip.push_back(shape.part(61));
			top_lip.push_back(shape.part(60));
			face_landmarks["top_lip"] = top_lip;
			for (auto i = 54; i < 60; i++)
				bottom_lip.push_back(shape.part(i));
			bottom_lip.push_back(shape.part(48));
			bottom_lip.push_back(shape.part(60));
			bottom_lip.push_back(shape.part(67));
			bottom_lip.push_back(shape.part(66));
			bottom_lip.push_back(shape.part(65));
			bottom_lip.push_back(shape.part(64));
			face_landmarks["bottom_lip"] = bottom_lip;
		}
		else
		{
			face_landmarks["nose_tip"] =
				std::vector<dlib::point>{shape.part(4)};
			face_landmarks["left_eye"] =
				std::vector<dlib::point>(shape.part(2), shape.part(4));
			face_landmarks["right_eye"] =
				std::vector<dlib::point>(shape.part(0), shape.part(2));
		}
		landmarks.push_back(face_landmarks);
	}
	return landmarks;
}
