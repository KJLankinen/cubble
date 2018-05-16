//-*- C++ -*-
#pragma once

#define CUBBLE_XSTRINGIFY(s) CUBBLE_STRINGIFY(s)
#define CUBBLE_STRINGIFY(s) #s
#define CUBBLE_PARAMETER(read, j, param)	\
    do						\
    {						\
	if (read)				\
	{					\
	    param = j[#param];			\
	    std::string	s(#param);		\
	    s += "Expl";			\
	    std::cout << j[s] << ": " << param;	\
	    std::cout << std::endl;		\
	}					\
	else					\
	{					\
	    j[#param] = param;			\
	}					\
    }						\
    while(0)
