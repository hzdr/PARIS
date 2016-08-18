#ifndef DDAFA_EXCEPTION_H_
#define DDAFA_EXCEPTION_H_

#include <exception>
#include <stdexcept>

namespace ddafa
{
    class stage_construction_error : public std::runtime_error
    {
        public:
            using std::runtime_error::runtime_error;
    };

    class stage_runtime_error : public std::runtime_error
    {
        public:
            using std::runtime_error::runtime_error;
    };
}

#endif /* EXCEPTION_H_ */
