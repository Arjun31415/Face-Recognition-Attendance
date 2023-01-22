#include "models/models.hpp"
#include <bits/stdc++.h>
#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/string.h>
using namespace dlib;
using namespace std;

int main(int argc, char **argv)
{
	Models m = Models();
	m.pose_predictor_model_location();
	return 0;
}
