/**
 * @file models.cpp
 * @author Arjun31415
 * @brief functions that return the path to models
 */

#include "models.hpp"
/**
 * @brief returns the path to the 68 point shape predictor model
 *
 * @return path to shape_predictor_68_face_landmarks.dat
 */
std::string Model::pose_predictor_model_location()
{
	return (std::filesystem::path(__FILE__).parent_path() / "model_assets" /
			"shape_predictor_68_face_landmarks.dat")
		.string();
}

/**
 * @brief returns the path to the 5 point shape predictor model
 *
 * @return path to shape_predictor_5_landmarks.dat
 */
std::string Model::pose_predictor_five_point_model_location()
{
	return (std::filesystem::path(__FILE__).parent_path() / "model_assets" /
			"shape_predictor_5_face_landmarks.dat")
		.string();
}

/**
 * @brief returns the path to the face recognition model
 *
 * @return path to dlib_Face_recognition_resnet_model_v1.dat
 */
std::string Model::face_recognition_model_location()
{
	return (std::filesystem::path(__FILE__).parent_path() / "model_assets" /
			"dlib_face_recognition_resnet_model_v1.dat")
		.string();
}

/**
 * @brief returns the path to the cnn face detector model
 *
 * @return path to mmod_human_face_detector.dat
 */
std::string Model::cnn_face_detector_model_location()
{
	return (std::filesystem::path(__FILE__).parent_path() / "model_assets" /
			"mmod_human_face_detector.dat")
		.string();
}
