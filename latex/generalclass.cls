%%%% generalclass.cls %%%%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{generalclass}[2012/04/19 v0.1 General class (DNJ ELS)]

\RequirePackage{etoolbox}

\newcommand*\DeclareClassList[2]{%
    \def\@clsopt@select{#1}%-------------------- default class
    \edef\@clsopt@list{\zap@space#2 \@empty}%--- list of classes
    \@for\@clsopt@item:=\@clsopt@list\do{%
        \newbool{@clsopt@test@\@clsopt@item}%
        \csedef{@clsopt@def@\@clsopt@item}{%
            \def\noexpand\@clsopt@select{\@clsopt@item}}}
    \edef\@currclass@opt{\@ptionlist{\@currname.\@currext}}%
    \@for\CurrentOption:=\@currclass@opt\do{%--- Go through documents options
        \ifcsdef{@clsopt@def@\CurrentOption}{%
            \csuse{@clsopt@def@\CurrentOption}%
            \csdef{ds@\CurrentOption}{}%
            \@use@ption}{}}}

\newcommand*\ThisClass{}
\newcommand\SetClassOption{%
    \edef\@clsopt@temp{%
        \noexpand\booltrue{@clsopt@test@\@clsopt@select}%
        \def\noexpand\ThisClass{\@clsopt@select}}%
    \@clsopt@temp}

\newcommand\IfClass[3]{%
    \ifbool{@clsopt@test@#1}{#2}{#3}}

%----------------------------------------------------------------------
\providecommand*\ClassDefault{article}                   % Edit 1
\providecommand*\ClassList{article,report,book,letter}   % Edit 1
\DeclareClassList{\ClassDefault}{\ClassList}             % Edit 1 
\SetClassOption

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\ThisClass}}
\ProcessOptions\relax
\LoadClass{\ThisClass}
\endinput