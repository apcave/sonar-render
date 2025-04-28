#ifndef _GLOBALS_C
#define _GLOBALS_C

enum ObjectType
{
    OBJECT_TYPE_TARGET = 0,
    OBJECT_TYPE_SOURCE,
    OBJECT_TYPE_FIELD
};

extern ObjectType safeConvertToEnum(int value);

#endif