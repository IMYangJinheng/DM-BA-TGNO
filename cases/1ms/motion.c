#include "udf.h"

DEFINE_CG_MOTION(mymotion, dt, vel, omega, time, dtime)
{
    if (time <= 0.2)
    {
        vel[0] = 1; 
        vel[1] = 0.0;
        vel[2] = 0.0;
    }
    else 
    {
        vel[0] = 0.0;
        vel[1] = 0.0;
        vel[2] = 0.0;
    }
}