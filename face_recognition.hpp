#pragma once
#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/string.h>

#include "models.hpp"

using namespace dlib;
template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<
	con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<
	con<1, 9, 9, 1, 1,
		rcon5<rcon5<
			rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

template <template <int, template <typename> class, int, typename> class block,
		  int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block,
		  int N, template <typename> class BN, typename SUBNET>
using residual_down =
	add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block =
	BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<
	128,
	avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<
		3, 3, 2, 2,
		relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

class FaceRecognition
{
	static inline const dlib::frontal_face_detector face_detector =
		dlib::get_frontal_face_detector();
	static inline std::string predictor_68_point_model =
		Model::pose_predictor_model_location();
	static inline std::string predictor_5_point_model =
		Model::pose_predictor_five_point_model_location();
	static inline std::string cnn_face_detection_model =
		Model::cnn_face_detector_model_location();
	static inline std::string face_recognition_model =
		Model::face_recognition_model_location();

	dlib::shape_predictor pose_predictor_68_point;
	dlib::shape_predictor pose_predictor_5_point;
	net_type cnn_face_detector;
	anet_type face_encoder;
	std::vector<matrix<float, 0, 1>> known_face_descriptors;
	std::vector<std::string> known_face_names;
	static std::tuple<long, long, long, long> _rect_to_css(dlib::rectangle rect)
	{
		return std::make_tuple(rect.top(), rect.right(), rect.bottom(),
							   rect.left());
	}
	static dlib::rectangle _css_to_rect(std::tuple<long, long, long, long> css)
	{
		return dlib::rectangle(std::get<3>(css), std::get<0>(css),
							   std::get<1>(css), std::get<2>(css));
	}
	static std::tuple<long, long, long, long>
	_trim_css_to_bounds(std::tuple<long, long, long, long> css,
						dlib::matrix<dlib::rgb_pixel> &img);

  public:
	FaceRecognition(void)
	{
		std::cout << predictor_5_point_model << "\n";
		dlib::deserialize(predictor_68_point_model) >> pose_predictor_68_point;
		dlib::deserialize(predictor_5_point_model) >> pose_predictor_5_point;
		dlib::deserialize(cnn_face_detection_model) >> cnn_face_detector;
		dlib::deserialize(face_recognition_model) >> face_encoder;
		return;
	}
	void _raw_face_locations(dlib::matrix<dlib::rgb_pixel> &img,
							 std::pair<int, int> res,
							 std::vector<dlib::mmod_rect> &,
							 std::string model = "cnn");
	void _batched_raw_face_locations(
		std::vector<dlib::matrix<dlib::rgb_pixel>> &img,
		std::pair<int, int> res,
		std::vector<std::vector<dlib::mmod_rect>> &face_locations,
		std::string model);
	void recognize_faces(matrix<rgb_pixel> &img,
						 std::vector<dlib::mmod_rect> &faces,
						 std::vector<dlib::mmod_rect> &overlay,std::vector<std::string>& names);
	void _get_image_files_in_directory(
		const std::filesystem::path &known_folder,
		std::vector<std::pair<std::string, std::string>> &);
	void scan_known_people(const std::filesystem::path &known_folder,
						   const std::pair<int, int> &res);
};
