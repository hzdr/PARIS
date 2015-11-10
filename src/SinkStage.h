/*
 * SinkStage.h
 *
 *  Created on: 10.11.2015
 *      Author: Jan Stephan
 *
 *      The SinkStage is the last Stage in the pipeline. It receives Image objects and saves them
 *      to the given path / volume.
 */

#ifndef SINKSTAGE_H_
#define SINKSTAGE_H_

#include <cstddef>
#include <string>

#include "Image.h"
#include "InputSide.h"

template <class ImageHandler>
class SinkStage : public ImageHandler, InputSide<Image>
{
	public:
		/* Constructs a SinkStage that writes to a directory */
		SinkStage(std::string path, std::string prefix)
		: use_volume_{false}
		{
			// TODO: Create target directory
		}

		/* Constructs a SinkStage that writes to a volume */
		SinkStage(std::string volume_path)
		: use_volume_{true}, path_{volume_path}
		{
		}

		/* Starts the stage */
		void start()
		{
			std::size_t index = 0;
			while(!queue_.empty())
			{
				// TODO: Create valid file paths for multiple images
				auto image = queue_.take();
				if(image.valid())
				{
					if(use_volume_)
						ImageHandler::saveToVolume(image, path_, index);
					else
						ImageHandler::saveImage(image, path_);
				}
				++index;
			}
		}

	private:
		bool use_volume_;
		std::string path_;
};


#endif /* SINKSTAGE_H_ */
