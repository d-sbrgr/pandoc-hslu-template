FROM pandoc/latex:latest-alpine
RUN apk add --no-cache \
    python3 \
    py3-pip \
    fontconfig \
    ghostscript

RUN tlmgr install datetime fmtcount xkeyval makecell xltabular ltablex tablefootnote xifthen lastpage \
enumitem titlesec lm lm-math tocloft adjustbox comment glossaries pdfx everyshi xmpincl siunitx tikzscale \
xstring epstopdf-pkg

RUN fc-cache -fv

WORKDIR /root

ENV HISTFILE=/dev/null
ENTRYPOINT ["/bin/sh"]