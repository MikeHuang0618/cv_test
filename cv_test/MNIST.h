#pragma once
#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct MNISTImageFileHeader
{
	unsigned char MagicNumber[4];
	unsigned char NumberOfImages[4];
	unsigned char NumberOfRows[4];
	unsigned char NumberOfColums[4];
};

struct MNISTLabelFileHeader
{
	unsigned char MagicNumber[4];
	unsigned char NumberOfLabels[4];
};

const int MAGICNUMBEROFIMAGE = 2051;
const int MAGICNUMBEROFLABEL = 2049;

int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray);

bool IsImageDataFile(unsigned char* MagicNumber, int LengthOfArray);

bool IsLabelDataFile(unsigned char* MagicNumber, int LengthOfArray);

Mat ReadData(fstream& DataFile, int NumberOfData, int DataSizeInBytes);

Mat ReadImageData(fstream& ImageDataFile, int NumberOfImages);

Mat ReadLabelData(fstream& LabelDataFile, int NumberOfLabel);

Mat ReadImages(string& FileName);

Mat ReadLabels(string& FileName);

#endif // !MNIST_H
