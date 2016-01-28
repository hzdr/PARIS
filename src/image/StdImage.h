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
		template <typename Data, class Allocator = std::allocator<Data>, class Deleter = std::default_delete<Data>>
		class StdImage : public Allocator
		{
			public:
				using value_type = Data;
				using deleter_type = Deleter;
				using allocator_type = Allocator;

			public:
				std::unique_ptr<value_type, deleter_type> allocate(std::size_t size)
				{
					return std::unique_ptr<value_type, deleter_type>(Allocator::allocate(size * sizeof(value_type)));
				}

				void copy(const value_type* src, value_type* dest, std::size_t size)
				{
					std::copy(src, src + size, dest);
				}

			protected:
				~StdImage()
				{
				}
		};
	}
}


#endif /* STDIMAGE_H_ */
