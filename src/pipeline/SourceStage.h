/*
 * SourceStage.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The SourceStage uses its ImageHandler policy to load images from the specified path. It will then
 *      forward those images to the next pipeline stage.
 */

#ifndef SOURCESTAGE_H_
#define SOURCESTAGE_H_

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include "../common/Filesystem.h"
#include "../image/Image.h"

#include "OutputSide.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class ImageLoader>
		class SourceStage : public ImageLoader
						  , public OutputSide<ddafa::image::Image<float, typename ImageLoader::image_type>>
		{
			public:
				using output_type = ddafa::image::Image<float, typename ImageLoader::image_type>;

			public:
				SourceStage(std::string path)
				: OutputSide<output_type>(), ImageLoader(), path_{path}
				{
				}

				void run()
				{
					std::vector<std::string> paths = ddafa::common::readDirectory(path_);
					for(auto&& path : paths)
					{
						output_type img = ImageLoader::template loadImage<float>(path);
						if(img.valid())
							this->output(std::move(img));
						else
							BOOST_LOG_TRIVIAL(warning) << "SourceStage: Skipping invalid file " << path;
					}

					// all images loaded, send poisonous pill
					BOOST_LOG_TRIVIAL(debug) << "SourceStage: Loading complete, sending poisonous pill";
					this->output(output_type());
				}

			private:
				std::string path_;
		};
	}
}


#endif /* SOURCESTAGE_H_ */
