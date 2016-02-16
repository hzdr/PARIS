/*
 * SinkStage.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The SinkStage uses its ImageHandler policy to save images or volumes that are being passed from
 *      previous pipeline stages.
 */

#ifndef SINKSTAGE_H_
#define SINKSTAGE_H_

#include <cstdint>
#include <string>
#include <utility>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include "../common/Filesystem.h"
#include "../image/Image.h"

#include "InputSide.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class ImageSaver>
		class SinkStage : public ImageSaver
						, public InputSide<ddafa::image::Image<float, typename ImageSaver::image_type>>
		{
			public:
				using input_type = ddafa::image::Image<float, typename ImageSaver::image_type>;

			public:
				SinkStage(const std::string& path, const std::string& prefix)
				: ImageSaver(), InputSide<input_type>(), path_{path}, prefix_{prefix}
				{
					bool created = ddafa::common::createDirectory(path_);
					if(!created)
						BOOST_LOG_TRIVIAL(fatal) << "SinkStage: Could not create target directory at " << path;

					if(path_.back() != '/')
						path_.append("/");
				}

				auto run() -> void
				{
					auto counter = 0;
					while(true)
					{
						auto img = this->input_queue_.take();
						if(img.valid())
						{
							ImageSaver::template saveImage<float>(std::move(img), path_ + prefix_ + std::to_string(counter));
							++counter;
						}
						else
						{
							BOOST_LOG_TRIVIAL(debug) << "SinkStage: Poisonous pill arrived, terminating.";
							break; // poisonous pill
						}
					}
				}

			private:
				std::string path_;
				std::string prefix_;
		};
	}
}


#endif /* SINKSTAGE_H_ */
