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

#include <string>
#include <utility>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

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
				: InputSide<input_type>(), ImageSaver(), target_dir_{path}, prefix_{prefix}
				{
				}

				void run()
				{
					while(true)
					{
						input_type img = this->input_queue_.take();
						if(img.valid())
							ImageSaver::template saveImage<float>(std::move(img), "/media/HDD1/Feldkamp/out.tif");
						else
						{
							BOOST_LOG_TRIVIAL(debug) << "SinkStage: Poisonous pill arrived, terminating.";
							break; // poisonous pill
						}

					}
				}

			private:
				std::string target_dir_;
				std::string prefix_;
		};
	}
}


#endif /* SINKSTAGE_H_ */
