/*
 * Image.h
 *
 *  Created on: 05.11.2015
 *      Author: Jan Stephan
 *
 *      Image class that holds a pointer to the concrete image data
 *      The individual objects are usually created by the ImageHandler and its corresponding handler policies.
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "StdImage.h"

namespace ddafa
{
	namespace image
	{
		template <typename Data, class Implementation>
		class Image : public Implementation
		{
			public:
				/*
				 * Constructs an empty image.
				 */
				Image() noexcept
				: width_{0}, height_{0}, data_{nullptr}, valid_{false}
				{
				}

				/*
				 * Constructs an image with the given dimensions. If img_data is a nullptr the data_ member will
				 * be allocated and all values will be initialized to zero. If img_data is not a nullptr the
				 * Image object will own the pointer that gets passed to it. In every case valid_ will be set to
				 * true after construction.
				 */
				Image(std::uint32_t img_width, std::uint32_t img_height,
						std::unique_ptr<Data, typename Implementation::deleter_type>&& img_data = nullptr)
				: width_{img_width}, height_{img_height}, data_{std::move(img_data)}, valid_{true}
				{
					if(data_ == nullptr)
							data_ = Implementation::allocate(width_ * height_);
				}

				/*
				 * Copy constructor
				 */
				Image(const Image& other)
				: Implementation(other)
				, width_{other.width_}, height_{other.height_}, valid_{other.valid_}
				{
					if(other.data_ == nullptr)
						data_ = nullptr;
					else
					{
						data_ = Implementation::allocate(width_ * height_);
						Implementation::copy(other.data_.get(), data_.get(), (width_ * height_));
					}
				}

				/*
				 * Copy assignment operator
				 */
				Image& operator=(const Image& rhs)
				{
					width_ = rhs.width_;
					height_ = rhs.height_;
					valid_ = rhs.valid_;

					if(rhs.data_ == nullptr)
						data_ = nullptr;
					else
					{
						data_.reset(nullptr); // delete old content if any
						data_ = Implementation::allocate(width_ * height_);
						Implementation::copy(rhs.data_.get(), data_.get(), (width_ * height_));
					}

					Implementation::operator=(rhs);
					return *this;
				}

				/*
				 * Move constructor
				 */
				Image(Image&& other) noexcept
				: Implementation(std::move(other))
				, width_{other.width_}, height_{other.height_}, data_{std::move(other.data_)}
				, valid_{other.valid_}
				{
					other.valid_ = false; // invalid after we moved its data
				}

				/*
				 * Move assignment operator
				 */
				Image& operator=(Image&& rhs) noexcept
				{
					width_ = rhs.width_;
					height_ = rhs.height_;
					data_ = std::move(rhs.data_);
					valid_ = rhs.valid_;

					Implementation::operator=(std::move(rhs));

					rhs.valid_ = false;
					return *this;
				}

				/*
				 * returns the image's width
				 */
				std::uint32_t width() const noexcept
				{
					return width_;
				}

				/*
				 * returns the image's height
				 */
				std::uint32_t height() const noexcept
				{
					return height_;
				}

				/*
				 * returns a non-owning pointer to the data. Do not delete this pointer as the Image object will take
				 * care of the memory.
				 */
				Data* data() const noexcept
				{
					return data_.get();
				}

				/*
				 * returns the validness of the image
				 */
				bool valid() const noexcept
				{
					return valid_;
				}

			private:
				std::uint32_t width_;
				std::uint32_t height_;
				std::unique_ptr<Data, typename Implementation::deleter_type> data_;
				bool valid_;
		};
	}
}



#endif /* IMAGE_H_ */
