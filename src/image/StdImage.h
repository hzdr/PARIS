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
		template <typename Data>
		class StdImage
		{
			public:
				using deleter_type = std::default_delete<Data>;

			public:
				std::unique_ptr<Data> allocate(std::size_t size)
				{
					return std::unique_ptr<Data>(new Data[size]);
				}

				void copy(const Data* src, Data* dest, std::size_t size)
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
