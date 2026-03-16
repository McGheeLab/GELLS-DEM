% pnas-new.cls — Simplified PNAS-style article class (XeTeX compatible)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pnas-new}[2024/01/01 Simplified PNAS article class]

\LoadClass[9pt,twoside]{extarticle}

% --- Font setup (XeTeX compatible, macOS system fonts) ---
\RequirePackage{fontspec}
\setmainfont{Times New Roman}[
  BoldFont = Times New Roman Bold,
  ItalicFont = Times New Roman Italic,
  BoldItalicFont = Times New Roman Bold Italic,
]
\setsansfont{Helvetica}
\setmonofont{Courier}

\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{enumitem}
\RequirePackage{xcolor}
\RequirePackage{etoolbox}
\RequirePackage{url}
\RequirePackage{lastpage}

% --- Page geometry (PNAS: 8.5 x 11, narrow margins for two-column) ---
\geometry{
  paperwidth=8.5in,
  paperheight=11in,
  left=0.75in,
  right=0.75in,
  top=0.75in,
  bottom=0.9in,
  columnsep=0.33in,
  headheight=14pt
}

% --- Colors ---
\definecolor{pnasblue}{RGB}{0,51,153}
\definecolor{pnasgray}{RGB}{100,100,100}
\definecolor{pnaslightgray}{RGB}{200,200,200}

% --- Header / Footer ---
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[LE,RO]{\footnotesize\thepage}
\fancypagestyle{firststyle}{%
  \fancyhf{}
  \fancyfoot[R]{\footnotesize\thepage}
}

% --- Section formatting ---
\titleformat{\section}
  {\normalfont\bfseries\sffamily\large\color{pnasblue}}
  {}
  {0em}
  {}
\titleformat{\subsection}
  {\normalfont\bfseries\sffamily\normalsize}
  {}
  {0em}
  {}
\titlespacing*{\section}{0pt}{12pt plus 2pt minus 2pt}{4pt plus 1pt minus 1pt}
\titlespacing*{\subsection}{0pt}{8pt plus 2pt minus 1pt}{2pt plus 1pt minus 1pt}

% --- Caption formatting ---
\captionsetup{
  font={small,sf},
  labelfont={bf},
  labelsep=period,
  justification=justified,
  singlelinecheck=false
}

% --- Metadata commands (all store content for \maketitle) ---
\newcommand{\significancestatement}[1]{\gdef\@significance{#1}}
\newcommand{\authorcontributions}[1]{\gdef\@authorcontributions{#1}}
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}
\newcommand{\leadauthor}[1]{\gdef\@leadauthor{#1}}
\newcommand{\correspondingauthor}[1]{\gdef\@correspondingauthor{#1}}
\newcommand{\dates}[1]{\gdef\@dates{#1}}
\long\def\pnasabstract#1{\gdef\@pnasabstract{#1}}

% Affiliation support
\newcommand{\affil}[2][]{%
  \expandafter\gdef\csname affil@#1\endcsname{#2}%
  \g@addto@macro\@affiliations{%
    \textsuperscript{#1}\,#2\\
  }%
}
\gdef\@affiliations{}
\gdef\@significance{}
\gdef\@authorcontributions{}
\gdef\@keywords{}
\gdef\@leadauthor{}
\gdef\@correspondingauthor{}
\gdef\@dates{}
\gdef\@pnasabstract{}

% --- Title formatting ---
\renewcommand{\maketitle}{%
  \twocolumn[%
    \begin{@twocolumnfalse}
    \vspace*{-0.5cm}
    {\raggedright
      {\fontsize{18}{22}\selectfont\bfseries\sffamily\color{pnasblue}\@title\par}
      \vspace{8pt}
      {\normalsize\@author\par}
      \vspace{4pt}
      {\footnotesize\color{pnasgray}\@affiliations\par}
      \ifx\@correspondingauthor\empty\else
        \vspace{2pt}
        {\footnotesize\color{pnasgray}\@correspondingauthor\par}
      \fi
      \ifx\@dates\empty\else
        \vspace{2pt}
        {\footnotesize\color{pnasgray}\@dates\par}
      \fi
    }
    \vspace{10pt}
    \hrule height 0.5pt
    \vspace{8pt}
    % Significance
    \ifx\@significance\empty\else
      {\small\sffamily\bfseries\color{pnasblue}Significance\par}
      \vspace{2pt}
      {\small\@significance\par}
      \vspace{6pt}
    \fi
    % Abstract
    \ifx\@pnasabstract\empty\else
      {\small\sffamily\bfseries\color{pnasblue}Abstract\par}
      \vspace{2pt}
      {\small\@pnasabstract\par}
      \vspace{6pt}
    \fi
    % Keywords
    \ifx\@keywords\empty\else
      {\footnotesize\sffamily\bfseries Keywords: \normalfont\@keywords\par}
      \vspace{4pt}
    \fi
    % Author contributions
    \ifx\@authorcontributions\empty\else
      {\footnotesize\sffamily\bfseries Author contributions: \normalfont\@authorcontributions\par}
      \vspace{4pt}
    \fi
    \hrule height 0.5pt
    \vspace{12pt}
    \end{@twocolumnfalse}
  ]%
}

% --- Compact lists ---
\setlist{nosep,leftmargin=1.5em}

% --- Paragraph spacing ---
\setlength{\parskip}{0pt plus 1pt}
\setlength{\parindent}{1em}

% --- Float placement ---
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.8}

\endinput
