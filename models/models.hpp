#ifndef MODLES_LIBRARY
#define MODLES_LIBRARY

#include <filesystem>
#include <iostream>
#include <string>
class Model
{
  public:
	std::string pose_predictor_model_location();
	std::string pose_predictor_five_point_model_location();
	std::string face_recognition_model_location();
	std::string cnn_face_detector_model_location();
};
#endif
