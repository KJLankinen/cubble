//-*- C++ -*-

#define _XSTRINGIFY(s) _STRINGIFY(s)
#define _STRINGIFY(s) #s
#define _PARAMETERIZE(read, param, j)		\
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
