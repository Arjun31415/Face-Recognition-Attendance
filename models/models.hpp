#ifndef MODLES_LIBRARY
#define MODLES_LIBRARY

#include <filesystem>
#include <iostream>
#include <string>
class Model
{
	inline static const std::string MODELS_PATH =
		"/home/azazel/cppProjects/Face-Recognition-Attendance/models";

  public:
	static std::string pose_predictor_model_location();
	static std::string pose_predictor_five_point_model_location();
	static std::string face_recognition_model_location();
	static std::string cnn_face_detector_model_location();
};
#endif
