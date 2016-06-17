#!/bin/sh

awk 'BEGIN{print "#include <string>\n#include \"version.h\"\n\nnamespace ddafa\n{"} END{}' > ../src/version.cpp
git rev-parse HEAD | awk 'BEGIN{} {print "\tstd::string git_build_sha = \""$0"\";"} END {}' >> ../src/version.cpp
date --iso-8601=seconds | awk 'BEGIN{} {print "\tstd::string git_build_time = \""$0"\";"} END {}' >> ../src/version.cpp
awk 'BEGIN{print "}"} END{}' >> ../src/version.cpp