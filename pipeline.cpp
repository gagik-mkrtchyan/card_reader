#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <map>
#include <chrono>
#include <iostream>

using namespace torch::indexing;

// Bilinear interpolation resize
torch::Tensor bilinear_interpolate_vectorized(const torch::Tensor& input, std::pair<int, int> size) {
    int c = input.size(0);
    int h = input.size(1);
    int w = input.size(2);
    int new_h = size.first;
    int new_w = size.second;

    torch::Tensor resized_image = torch::zeros({c, new_h, new_w}, input.options());

    auto scale_y = torch::linspace(0, h - 1, new_h, torch::kFloat32);
    auto scale_x = torch::linspace(0, w - 1, new_w, torch::kFloat32);

    auto x0 = torch::floor(scale_x).to(torch::kLong);
    auto x1 = torch::clamp(x0 + 1, 0, w - 1);
    auto y0 = torch::floor(scale_y).to(torch::kLong);
    auto y1 = torch::clamp(y0 + 1, 0, h - 1);

    auto dx = scale_x - x0.to(torch::kFloat);
    auto dy = scale_y - y0.to(torch::kFloat);

    for (int i = 0; i < c; ++i) {

        auto input_i = input[i];

        auto q11 = input_i.index({y0, Slice()});
        q11 = q11.index({Slice(), x0});

        auto q12 = input_i.index({y0, Slice()});
        q12 = q12.index({Slice(), x1});

        auto q21 = input_i.index({y1, Slice()});
        q21 = q21.index({Slice(), x0});

        auto q22 = input_i.index({y1, Slice()});
        q22 = q22.index({Slice(), x1});

        auto top = q11 * (1 - dx) + q12 * dx;
        auto bottom = q21 * (1 - dx) + q22 * dx;

        resized_image[i] = top * (1 - dy).unsqueeze(1) + bottom * dy.unsqueeze(1);
    }

    return resized_image.unsqueeze(0); // Add batch dimension
}

// Convert bounding box from xywh to xyxy
torch::Tensor xywh2xyxy(const torch::Tensor& x) {

    auto y = torch::empty_like(x);
    auto xy = x.index({Slice(), Slice(), Slice(0, 2)});
    auto wh = x.index({Slice(), Slice(), Slice(2, 4)}) / 2;

    y.index_put_({Slice(), Slice(), Slice(0, 2)}, xy - wh);
    y.index_put_({Slice(), Slice(), Slice(2, 4)}, xy + wh);
    return y;
}

// Compute Intersection over Union (IoU)
float iou(const torch::Tensor& box_a, const torch::Tensor& box_b) {
    float x1 = std::max(box_a[0].item<float>(), box_b[0].item<float>());
    float y1 = std::max(box_a[1].item<float>(), box_b[1].item<float>());
    float x2 = std::min(box_a[2].item<float>(), box_b[2].item<float>());
    float y2 = std::min(box_a[3].item<float>(), box_b[3].item<float>());

    float intersection_area = std::max(x2 - x1, 0.0f) * std::max(y2 - y1, 0.0f);
    float box_a_area = (box_a[2].item<float>() - box_a[0].item<float>()) * (box_a[3].item<float>() - box_a[1].item<float>());
    float box_b_area = (box_b[2].item<float>() - box_b[0].item<float>()) * (box_b[3].item<float>() - box_b[1].item<float>());

    return intersection_area / (box_a_area + box_b_area - intersection_area);
}

torch::Tensor nms(const torch::Tensor &boxes, const torch::Tensor &scores, float iou_thres) {

    auto keep_indices = scores >= iou_thres;

    auto filtered_boxes = boxes.index({keep_indices});
    auto filtered_scores = scores.index({keep_indices});

    auto sorted_indices = std::get<1>(filtered_scores.sort(0, /* descending=*/true));
    auto sorted_scores = std::get<0>(torch::sort(filtered_scores, /*dim=*/0, /*descending=*/true));

    filtered_boxes = filtered_boxes.index({sorted_indices});
    filtered_scores = sorted_scores;

    std::vector<torch::Tensor> F;


    while (filtered_boxes.size(0) != 0) {

        auto current_box = filtered_boxes.index({0});
        F.push_back(current_box);
        std::vector<float> ious;

        for (int i = 1; i < filtered_boxes.size(0); ++i) {
            float iou_value = iou(current_box, filtered_boxes.index({i}));
            ious.push_back(iou_value);
        }

        auto ious_tensor = torch::tensor(ious, torch::kFloat32);
        auto keep_indices = ious_tensor < iou_thres;

        filtered_boxes = filtered_boxes.index({Slice(1, None)});
        filtered_boxes = filtered_boxes.index({keep_indices});
        filtered_scores = filtered_scores.index({Slice(1, None)});
        filtered_scores = filtered_scores.index({keep_indices});
    }

    auto result = torch::stack(F);

    // Find rows in the original boxes that match the result boxes
    auto rows_all_true = torch::all(torch::isin(boxes, result), /*dim=*/1);
    auto indices_rows_all_true = torch::nonzero(rows_all_true);

    return indices_rows_all_true;
}


// Non-Maximum Suppression (NMS)
torch::Tensor non_max_suppression(
    const torch::Tensor &prediction,
    int nc,
    float conf_thres=0.25,
    float iou_thres=0.45,
    int max_det = 300,
    int max_nms = 30000,
    int max_wh = 7680
) {

    int bs = prediction.size(0);
    int nm = prediction.size(1) - nc - 4;
    int mi = 4 + nc;
    auto xc = prediction.index({Slice(), Slice(4, mi)}).amax(1) > conf_thres;

    auto prediction_transposed = prediction.transpose(-1, -2);
    auto sliced_tensor = prediction_transposed.index({Slice(), Slice(), Slice(0, 4)});


    auto res = xywh2xyxy(sliced_tensor);
    prediction_transposed.index_put_({Slice(), Slice(), Slice(0, 4)}, res);
    std::vector<torch::Tensor> output(bs, torch::zeros({0, 6 + nm}, prediction.device()));

    for (int xi = 0; xi < prediction.size(0); ++xi) {
        auto x = prediction_transposed[xi].index({xc[xi]});

        if (!x.size(0)) {
            continue;
        }
        auto split_tensors = torch::split(x, {4, nc, nm}, /*dim=*/1);
        auto box = split_tensors[0];
        auto cls = split_tensors[1];
        auto mask = split_tensors[2];
        auto result = torch::max(cls, /*dim=*/1, /*keepdim=*/true);

        auto conf = std::get<0>(result);
        auto j = std::get<1>(result);
        auto j_float = j.to(torch::kFloat);
        auto x_conc = torch::cat({box, conf, j_float, mask}, 1);

        auto conf_flat = conf.view({-1});
        auto mask_greater_than_thresh = conf_flat > conf_thres;
        auto x_filtered = x_conc.index({mask_greater_than_thresh});

        int n = x_filtered.size(0);
        if(!n) {
            continue;
        }

        if(n > max_nms) {
            auto sorted = torch::sort(x_filtered.index({Slice(), 4}), /*dim=*/0, /*descending=*/true);
            auto indices = std::get<1>(sorted);

            // Keep the top max_nms entries
            auto top_indices = indices.index({Slice(0, max_nms)});
            x_filtered = x_filtered.index({top_indices});
        }

        auto c = x_filtered.index({Slice(), Slice(5, 6)}) * max_wh;
        auto scores = x_filtered.index({Slice(), 4});
        auto boxes = x_filtered.index({Slice(), Slice(0, 4)}) + c;
        auto i = nms(boxes, scores, iou_thres);
        i = i.index({Slice(0, max_det)});
        auto selected_boxes = x_filtered.index({i});
        selected_boxes  = selected_boxes.permute({1, 0, 2});
        selected_boxes = selected_boxes.squeeze(0);
        output[xi] = selected_boxes;
    }

    torch::Tensor combined_tensor = torch::stack(output, 0);

    return combined_tensor.squeeze(0);
}

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    // Load the image using OpenCV
    std::string imagePath = "example.jpeg";
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }
    torch::Tensor imgTensor = torch::from_blob(
        img.data,
        { img.rows, img.cols, img.channels() },
        torch::kByte
    );

    imgTensor = imgTensor.to(torch::kFloat32) / 255.0;
    imgTensor = imgTensor.permute({2, 0, 1});
    auto resizedImg = bilinear_interpolate_vectorized(imgTensor, {640, 640});

    torch::jit::script::Module first_model = torch::jit::load("s3://fastbank-ml-models-archive/OCR/Weights/first_model.pt");
    torch::Tensor first_model_results;
    {
        torch::NoGradGuard no_grad;
        first_model_results = first_model.forward({resizedImg}).toTensor();
    }

    auto post_process = non_max_suppression(first_model_results, 3);

    torch::Tensor mask = post_process.index({Slice(), 4}) >= 0.7;

    post_process = post_process.index({mask});
    auto post_process_sorted_indices = std::get<1>(torch::sort(post_process.index({Slice(), -1}), /*dim=*/0, /*descending=*/true));
    post_process = post_process.index({post_process_sorted_indices});

    if(post_process.sizes()[0] != 3){
	std::cerr << "Error: Model did not recognize all three fields!" << std::endl;
        return -1;
    }

    auto expire_date_img = post_process.index({0});
    auto cardholder_img = post_process.index({1});
    auto cardnumber_img = post_process.index({2});

    auto w1 = torch::ceil(expire_date_img.index({2})).item<int>() - torch::ceil(expire_date_img.index({0})).item<int>();
    auto w2 = torch::ceil(cardholder_img.index({2})).item<int>() - torch::ceil(cardholder_img.index({0})).item<int>();
    auto w3 = torch::ceil(cardnumber_img.index({2})).item<int>() - torch::ceil(cardnumber_img.index({0})).item<int>();

    auto cropped_img1 = resizedImg.index({
                                          Slice(),
                                          Slice(),
                                          Slice(torch::ceil(expire_date_img.index({1})).item<int>(),torch::ceil(expire_date_img.index({3})).item<int>()),
                                          Slice(torch::ceil(expire_date_img.index({0})).item<int>(),torch::ceil(expire_date_img.index({2})).item<int>())
    });

    auto cropped_img2 = resizedImg.index({
                                          Slice(),
                                          Slice(),
                                          Slice(torch::ceil(cardholder_img.index({1})).item<int>(),torch::ceil(cardholder_img.index({3})).item<int>()),
                                          Slice(torch::ceil(cardholder_img.index({0})).item<int>(),torch::ceil(cardholder_img.index({2})).item<int>())
    });

    auto cropped_img3 = resizedImg.index({
                                          Slice(),
                                          Slice(),
                                          Slice(torch::ceil(cardnumber_img.index({1})).item<int>(),torch::ceil(cardnumber_img.index({3})).item<int>()),
                                          Slice(torch::ceil(cardnumber_img.index({0})).item<int>(),torch::ceil(cardnumber_img.index({2})).item<int>())
    });

    int max_w = std::max({w1, w2, w3});

    auto img1_padding = torch::zeros({1, 3, cropped_img1.sizes()[2], max_w-w1});
    auto img2_padding = torch::zeros({1, 3, cropped_img2.sizes()[2], max_w-w2});
    auto img3_padding = torch::zeros({1, 3, cropped_img3.sizes()[2], max_w-w3});

    expire_date_img = torch::cat({cropped_img1, img1_padding}, 3);
    cardholder_img = torch::cat({cropped_img2, img2_padding}, 3);
    cardnumber_img = torch::cat({cropped_img3, img3_padding}, 3);

    auto stacked_img = torch::cat({cardholder_img, expire_date_img, cardnumber_img}, 2);
    auto stacked_resized = bilinear_interpolate_vectorized(stacked_img.squeeze(0), {320, 800});

    torch::jit::script::Module second_model = torch::jit::load("s3://fastbank-ml-models-archive/OCR/Weights/second_model.pt");
    torch::Tensor second_model_results;
    {
    	torch::NoGradGuard no_grad;
        second_model_results = second_model.forward({stacked_resized}).toTensor();
    }
    auto second_post_process = non_max_suppression(second_model_results, 36);

    auto second_post_process_indices = std::get<1>(torch::sort(second_post_process.index({Slice(), -1}), /*dim=*/0));
    auto data = second_post_process.index({second_post_process_indices});

    auto mask_cardholder = data.index({Slice(), -1}) > 9 & data.index({Slice(), 4}) >= 0.65;
    auto card_holder = data.index({mask_cardholder});

    card_holder = card_holder.index({std::get<1>(torch::sort(card_holder.index({Slice(), 0}), /*dim=*/0))});
    card_holder = card_holder.index({Slice(), -1});
    std::map<int, char> letter_map = {
        {10, 'A'},
        {11, 'B'},
        {12, 'C'},
        {13, 'D'},
        {14, 'E'},
        {15, 'F'},
        {16, 'G'},
        {17, 'H'},
        {18, 'I'},
        {19, 'J'},
        {20, 'K'},
        {21, 'L'},
        {22, 'M'},
        {23, 'N'},
        {24, 'O'},
        {25, 'P'},
        {26, 'Q'},
        {27, 'R'},
        {28, 'S'},
        {29, 'T'},
        {30, 'U'},
        {31, 'V'},
        {32, 'W'},
        {33, 'X'},
        {34, 'Y'},
        {35, 'Z'}
    };

    for(int i = 0; i < card_holder.sizes()[0]; i++){
		std::cout << letter_map[card_holder[i].item<int>()] << " ";
    }
    std::cout << '\n' << std::endl;

    auto mask_date_card_number = data.index({Slice(), -1}) <= 9 & data.index({Slice(), 4}) >= 0.5 & data.index({Slice(), 1}) >= 20;
    auto date_card_number = data.index({mask_date_card_number});
    date_card_number = data.index({std::get<1>(torch::sort(date_card_number.index({Slice(), 1}), /*dim=*/0))});

    auto mask_expire_date = date_card_number.index({Slice(), 1}) < date_card_number.index({0, 3})-10 & date_card_number.index({Slice(), 4}) >= 0.5;
    auto mask_card_number = date_card_number.index({Slice(), 1}) > date_card_number.index({0, 3})-10 & date_card_number.index({Slice(), 4}) >= 0.7;

    auto expire_date = date_card_number.index({mask_expire_date});
    auto card_number = date_card_number.index({mask_card_number});

    card_number = card_number.index({std::get<1>(torch::sort(card_number.index({Slice(), 0}), /*dim=*/0))});
    card_number = card_number.index({Slice(), -1});

    for(int i = 0; i < card_number.sizes()[0]; i++){
		std::cout << card_number[i].item<int>() << " ";
    }
    std::cout << '\n' << std::endl;

    expire_date = expire_date.index({std::get<1>(torch::sort(expire_date.index({Slice(), 0}), /*dim=*/0))});
    expire_date = expire_date.index({Slice(), -1});

    for(int i = 0; i < expire_date.sizes()[0]; i++){
		std::cout << expire_date[i].item<int>() << " ";
    }
    return 0;
}
