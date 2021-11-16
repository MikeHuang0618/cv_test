#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include<opencv2/opencv.hpp>

#include "MNIST.h"

using namespace cv;
using namespace std;

vector<int> Argmax(Mat x) {
	vector<int> res;
	for (int i = 0; i < x.rows; i++) {
		int maxIdx = 0;
		float maxNum = 0.0;
		for (int j = 0; j < x.cols; j++) {
			float tmp = x.at<float>(i, j);
			if (tmp > maxNum) {
				maxIdx = j;
				maxNum = tmp;
			}
		}
		res.push_back(maxIdx);
	}

	return res;
}

float Accuracy(Mat x, Mat y, string pbfile) {
	float count = 0.0;
	dnn::Net net = dnn::readNetFromTensorflow(pbfile);

	int size[] = { x.rows, 28, 28 };
	Mat imgs = Mat(3, size, CV_8UC1, x.data);

	Mat blob = dnn::blobFromImages(imgs, 1.0 / 255.0, Size(28, 28), Scalar(), false, false);
	net.setInput(blob);
	Mat pred = net.forward();

	vector<int> res = Argmax(pred);

	for (int i = 0; i < res.size(); i++) {
		if (*(y.ptr<int>(0) + i) == res[i]) {
			count = count + 1;
		}
	}

	return count / x.rows;
}

int main()
{
	string testLabelFile = "data/t10k-labels-idx1-ubyte";
	string testImageFile = "data/t10k-images-idx3-ubyte";

	string trainLabelFile = "data/train-labels-idx1-ubyte";
	string trainImageFile = "data/train-images-idx3-ubyte";

	Mat trainY = ReadLabels(trainLabelFile);
	Mat testY = ReadLabels(testLabelFile);

	Mat trainX = ReadImages(trainImageFile);
	Mat testX = ReadImages(testImageFile);

	testY.convertTo(testY, CV_32SC1);

	string pbfile = "models/frozen_graph.pb";
	float acc = Accuracy(testX, testY, pbfile);
	cout << acc;
	return 0;
}