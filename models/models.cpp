#include "models.hpp"
std::string Model::pose_predictor_model_location()
{
	return (std::filesystem::path(__FILE__).parent_path() / "model_assets" /
			"shape_predictor_68_face_landmarks.dat")
		.string();
}

std::string Model::pose_predictor_five_point_model_location()
{
    // std::cout<<std::filesystem::path(__FILE__).string()<<"\n";
	return (std::filesystem::path(__FILE__).parent_path() / "model_assets" /
			"shape_predictor_5_face_landmarks.dat")
		.string();
}

std::string Model::face_recognition_model_location()
{
	return (std::filesystem::path(__FILE__).parent_path() / "model_assets" /
			"dlib_face_recognition_resnet_model_v1.dat")
		.string();
}

std::string Model::cnn_face_detector_model_location()
{
	return (std::filesystem::path(__FILE__).parent_path() / "model_assets" /
			"mmod_human_face_detector.dat")
		.string();
}
