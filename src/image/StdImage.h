/*
 * StdImage.h
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      StdImage provides a standard implementation for the Image class, relying entirely on the STL.
 */

#ifndef STDIMAGE_H_
#define STDIMAGE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ddafa
{
	namespace impl
	{
		template <typename Data, class Allocator, class Deleter = std::default_delete<Data>>
		class StdImage : public Allocator
		{
			public:
				using value_type = Data;
				using deleter_type = Deleter;
				using allocator_type = Allocator;
				using size_type = std::size_t;

			public:
				auto allocate(size_type width, size_type height, size_type* pitch) -> std::unique_ptr<value_type, deleter_type>
				{
					return std::unique_ptr<value_type, deleter_type>(Allocator::allocate(width, height, pitch));
				}

				auto copy(const value_type* src, value_type* dest, size_type width, size_type height, size_type) -> void
				{
					std::copy(src, src + (width * height), dest);
				}

			protected:
				~StdImage() = default;

			protected:
				size_type pitch_;
		};
	}
}


#endif /* STDIMAGE_H_ */
