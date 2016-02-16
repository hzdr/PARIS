/*
 * ImageSaver.h
 *
 *  Created on: 26.01.2016
 *      Author: Jan Stephan
 */

#ifndef IMAGESAVER_H_
#define IMAGESAVER_H_

namespace ddafa
{
	namespace image
	{
		template <class Implementation>
		class ImageSaver : public Implementation
		{
			public:
				using image_type = typename Implementation::image_type;

			public:
				/*
				 * Saves an image to the given path.
				 */
				template <typename T>
				auto saveImage(Image<T, image_type>&& image, std::string path) -> void
				{
					Implementation::saveImage(std::forward<Image<T, image_type>&&>(image), path);
				}

				/*
				 * Saves an image into a volume at the given path.
				 */
				template <typename T>
				auto saveToVolume(Image<T, image_type>&& image, std::string path, std::size_t index) -> void
				{
					Implementation::saveToVolume(std::forward<Image<T, image_type>&&>(image),
													path, index);
				}

			protected:
				~ImageSaver() = default;

		};
	}
}



#endif /* IMAGESAVER_H_ */
