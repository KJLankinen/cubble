//-*- C++ -*-

#define _XSTRINGIFY(s) _STRINGIFY(s)
#define _STRINGIFY(s) #s
#define _PARAMETER(read, j, param)		\
    do						\
    {						\
	if (read)				\
	{					\
	    param = j[#param];			\
	}					\
	else					\
	{					\
	    j[#param] = param;			\
	}					\
    }						\
    while(0)
