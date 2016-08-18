#ifndef DDAFA_SOURCE_STAGE_H_
#define DDAFA_SOURCE_STAGE_H_

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "his_loader.h"
#include "metadata.h"

namespace ddafa
{
    class source_stage
    {
        public:
            using loader_type = his_loader;
            using output_type = std::pair<his_loader::smart_pointer, image_metadata>;

        public:
            source_stage(const std::string& dir);
            auto run() -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            loader_type loader_;
            std::function<void(output_type)> output_;
            std::vector<std::string> paths_;
    };
}



#endif /* DDAFA_SOURCE_STAGE_H_ */
