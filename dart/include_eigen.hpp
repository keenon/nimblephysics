#ifndef DART_EIGEN_
#define DART_EIGEN_

// To get Intellisense to work on ARM M1 Macs, until Intellisense adds support for ARM_NEON instruction sets
// See: https://github.com/microsoft/vscode-cpptools/issues/7413#issuecomment-827172897
// Underlying bug tracker: https://developercommunity.visualstudio.com/t/C-IntelliSense-doesnt-work-correctly/1408223
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <Eigen/Dense>

#endif