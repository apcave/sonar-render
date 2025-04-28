#include "Globals.h"
#include <stdexcept>
#include <iostream>

ObjectType safeConvertToEnum(int value)
{
    if (value >= static_cast<int>(ObjectType::OBJECT_TYPE_TARGET) &&
        value <= static_cast<int>(ObjectType::OBJECT_TYPE_FIELD))
    {
        return static_cast<ObjectType>(value);
    }
    else
    {
        throw std::invalid_argument("Invalid value for ObjectType enum");
    }
}