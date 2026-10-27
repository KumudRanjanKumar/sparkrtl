#!/bin/bash
# OPTIONAL: build Icarus Verilog 12 on a Mac without Homebrew (about 10-15 minutes).
# You only need this to run VerilogEval locally; Kaggle notebook 01 builds it for you.
# GNU m4, autoconf and a newer bison are built first because macOS ships old or no versions.
set -e
P="$HOME/iv12-tools"
mkdir -p "$P/src" && cd "$P/src"
export PATH="$P/bin:$PATH"
curl -sLO https://ftp.gnu.org/gnu/m4/m4-1.4.19.tar.gz && tar xzf m4-1.4.19.tar.gz
(cd m4-1.4.19 && ./configure --prefix="$P" && make -j8 && make install)
curl -sLO https://ftp.gnu.org/gnu/autoconf/autoconf-2.72.tar.gz && tar xzf autoconf-2.72.tar.gz
(cd autoconf-2.72 && ./configure --prefix="$P" && make -j8 && make install)
curl -sLO https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.gz && tar xzf bison-3.8.2.tar.gz
(cd bison-3.8.2 && ./configure --prefix="$P" && make -j8 && make install)
rm -rf iverilog && git clone -q --depth 1 -b v12-branch https://github.com/steveicarus/iverilog.git
(cd iverilog && sh ./autoconf.sh && ./configure --prefix="$P/iv12" && make -j8 && make install)
"$P/iv12/bin/iverilog" -V 2>&1 | head -1
echo "Use it with:  export RTLRL_IVERILOG=$P/iv12/bin/iverilog RTLRL_VVP=$P/iv12/bin/vvp"
