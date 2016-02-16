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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace ddafa
{
	namespace image
	{
		template <typename Data, class Implementation>
		class Image : public Implementation
		{
			public:
				using value_type = Data;
				using deleter_type = typename Implementation::deleter_type;
				using size_type = std::size_t;

			public:
				/*
				 * Constructs an empty image.
				 */
				Image() noexcept
				: width_{0}, height_{0}, pitch_{0}, data_{nullptr}, valid_{false}
				{
				}

				/*
				 * Constructs an image with the given dimensions. If img_data is a nullptr the data_ member will
				 * be allocated and all values will be initialized to zero. If img_data is not a nullptr the
				 * Image object will own the pointer that gets passed to it. In every case valid_ will be set to
				 * true after construction.
				 */
				Image(size_type img_width, size_type img_height,
						std::unique_ptr<value_type, deleter_type>&& img_data = nullptr)
				: width_{img_width}, height_{img_height}, data_{std::move(img_data)}, valid_{true}
				{
					if(data_ == nullptr)
							data_ = Implementation::allocate(width_, height_, &pitch_);
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
						data_ = Implementation::allocate(width_, height_, &pitch_);
						Implementation::copy(other.data_.get(), data_.get(), width_, height_, pitch_);
					}
				}

				/*
				 * Copy assignment operator
				 */
				auto operator=(const Image& rhs) -> Image&
				{
					width_ = rhs.width_;
					height_ = rhs.height_;
					valid_ = rhs.valid_;

					if(rhs.data_ == nullptr)
						data_ = nullptr;
					else
					{
						data_.reset(nullptr); // delete old content if any
						data_ = Implementation::allocate(width_, height_, &pitch_);
						Implementation::copy(rhs.data_.get(), data_.get(), width_, height_, pitch_);
					}

					Implementation::operator=(rhs);
					return *this;
				}

				/*
				 * Move constructor
				 */
				Image(Image&& other) noexcept
				: Implementation(std::move(other))
				, width_{other.width_}, height_{other.height_}, pitch_{other.pitch_}, data_{std::move(other.data_)}
				, valid_{other.valid_}
				{
					other.valid_ = false; // invalid after we moved its data
				}

				/*
				 * Move assignment operator
				 */
				auto operator=(Image&& rhs) noexcept -> Image&
				{
					width_ = rhs.width_;
					height_ = rhs.height_;
					pitch_ = rhs.pitch_;
					data_ = std::move(rhs.data_);
					valid_ = rhs.valid_;

					Implementation::operator=(std::move(rhs));

					rhs.valid_ = false;
					return *this;
				}

				/*
				 * returns the image's width
				 */
				auto width() const noexcept -> size_type
				{
					return width_;
				}

				/*
				 * returns the image's height
				 */
				auto height() const noexcept -> size_type
				{
					return height_;
				}

				/*
				 * returns a non-owning pointer to the data. Do not delete this pointer as the Image object will take
				 * care of the memory.
				 */
				auto data() const noexcept -> value_type*
				{
					return data_.get();
				}

				/*
				 * returns the validness of the image
				 */
				auto valid() const noexcept -> bool
				{
					return valid_;
				}

				auto pitch(size_type new_pitch) noexcept -> void
				{
					pitch_ = new_pitch;
				}

				auto pitch() const noexcept -> size_type
				{
					return pitch_;
				}

			private:
				size_type width_;
				size_type height_;
				size_type pitch_;
				std::unique_ptr<value_type, deleter_type> data_;
				bool valid_;
		};
	}
}



#endif /* IMAGE_H_ */
