//-*- C++ -*-

#define CUBBLE_XSTRINGIFY(s) CUBBLE_STRINGIFY(s)
#define CUBBLE_STRINGIFY(s) #s
#define CUBBLE_PARAMETER(read, j, param)	\
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
