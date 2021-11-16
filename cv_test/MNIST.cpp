#include "MNIST.h"

int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray) {
	if (LengthOfArray < 0) {
		return -1;
	}

	int result = static_cast<signed int>(array[0]);
	for (int i = 1; i < LengthOfArray; i++) {
		result = (result << 8) + array[i];
	}

	return result;
}

bool IsImageDataFile(unsigned char* MagicNumber, int LengthOfArray)
{
	int MagicNumberOfImage = ConvertCharArrayToInt(MagicNumber, LengthOfArray);
	if (MagicNumberOfImage == MAGICNUMBEROFIMAGE) {
		return true;
	}

	return false;
}

bool IsLabelDataFile(unsigned char* MagicNumber, int LengthOfArray)
{
	int MagicNumberOfLabel = ConvertCharArrayToInt(MagicNumber, LengthOfArray);
	if (MagicNumberOfLabel == MAGICNUMBEROFLABEL) {
		return true;
	}

	return false;
}

Mat ReadData(fstream& DataFile, int NumberOfData, int DataSizeInBytes)
{
	Mat DataMat;

	if (DataFile.is_open()) {
		int AllDataSizeInBytes = DataSizeInBytes * NumberOfData;
		char* TmpData = new char[AllDataSizeInBytes];
		DataFile.read((char*)TmpData, AllDataSizeInBytes);
		DataMat = Mat(NumberOfData, DataSizeInBytes, CV_8UC1, TmpData).clone();

		delete[] TmpData;
		DataFile.close();
	}

	return DataMat;
}

Mat ReadImageData(fstream& ImageDataFile, int NumberOfImages)
{
	int ImageSizeInBytes = 28 * 28;

	return ReadData(ImageDataFile, NumberOfImages, ImageSizeInBytes);
}

Mat ReadLabelData(fstream& LabelDataFile, int NumberOfLabel)
{
	int LabelSizeInBytes = 1;

	return ReadData(LabelDataFile, NumberOfLabel, LabelSizeInBytes);
}

Mat ReadImages(string& FileName)
{
	fstream File(FileName.c_str(), ios_base::in | ios_base::binary);

	if (!File.is_open()) {
		return Mat();
	}

	MNISTImageFileHeader FileHeader;
	File.read((char*)(&FileHeader), sizeof(FileHeader));

	if (!IsImageDataFile(FileHeader.MagicNumber, 4)) {
		return Mat();
	}

	int NumberOfImage = ConvertCharArrayToInt(FileHeader.NumberOfImages, 4);

	return ReadImageData(File, NumberOfImage);
}

Mat ReadLabels(string& FileName)
{
	fstream File(FileName.c_str(), ios_base::in | ios_base::binary);

	if (!File.is_open()) {
		return Mat();
	}

	MNISTLabelFileHeader FileHeader;
	File.read((char*)(&FileHeader), sizeof(FileHeader));

	if (!IsLabelDataFile(FileHeader.MagicNumber, 4)) {
		return Mat();
	}

	int NumberOfImage = ConvertCharArrayToInt(FileHeader.NumberOfLabels, 4);

	return ReadLabelData(File, NumberOfImage);
}
