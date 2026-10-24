// READ ONLY. No control, recovery, tool configuration or gripper calls.
#include <franka/robot.h>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv) {
  try {
    franka::Robot robot(argc>1?argv[1]:"172.16.0.2",franka::RealtimeConfig::kIgnore);
    const auto state=robot.readOnce();
    std::cout<<std::setprecision(17)<<state<<std::endl;
  } catch(const std::exception&e) {std::cerr<<e.what()<<std::endl;return 1;}
}
