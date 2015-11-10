/*
 * SourceStage.h
 *
 *  Created on: 10.11.2015
 *      Author: Jan Stephan
 *
 *      The SourceStage loads the Images from the given path and passes them to the next stage.
 */

#ifndef SOURCESTAGE_H_
#define SOURCESTAGE_H_

#include <string>
#include <vector>
#include <utility>

#include "Image.h"
#include "OutputSide.h"

template <class ImageHandler>
class SourceStage : public ImageHandler, public OutputSide<Image>
{
	public:
		SourceStage(std::string path)
		{
			// TODO: construct paths_
		}

		void start()
		{
			for(auto path : paths_)
			{
				auto image = ImageHandler::loadImage(path);
				if(image.valid())
					OutputSide::output(std::move(image));
			}
		}

	private:
		std::vector<std::string> paths_;
};



#endif /* SOURCESTAGE_H_ */
