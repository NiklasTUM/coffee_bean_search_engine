
## 1. Regular Expression Tutorial

In this tutorial, I will teach you all you need to know to be able to craft powerful time-saving regular expressions. I will start with the most basic concepts, so that you can follow this tutorial even if you know nothing at all about regular expressions yet. But I will not stop there. I will also explain how a regular expression engine works on the inside, and alert you at the consequences. This will help you to understand quickly why a particular regex does not do what you initially expected. It will save you lots of guesswork and head scratching when you need to write more complex regexes. 

## What Regular Expressions Are Exactly - Terminology

Basically, a regular expression is a pattern describing a certain amount of text. Their name comes from the mathematical theory on which they are based. But we will not dig into that. Since most people including myself are lazy to type, you will usually find the name abbreviated to regex or regexp. I prefer regex, because it is easy to pronounce the plural "regexes". In this book, regular expressions are printed between guillemots: 
«regex». They clearly separate the pattern from the surrounding text and punctuation. This first example is actually a perfectly valid regex. It is the most basic pattern, simply matching the literal text "regex". A "match" is the piece of text, or sequence of bytes or characters that pattern was found to correspond to by the regex processing software. Matches are indicated by double quotation marks, with the left one at the base of the line. 

«\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b» is a more complex pattern. It describes a series of letters, digits, dots, underscores, percentage signs and hyphens, followed by an at sign, followed by another series of letters, digits and hyphens, finally followed by a single dot and between two and four letters. In other words: this pattern describes an email address. With the above regular expression pattern, you can search through a text file to find email addresses, or verify if a given string looks like an email address. In this tutorial, I will use the term "string" to indicate the text that I am applying the regular expression to. I will indicate strings using regular double quotes. The term "string" or "character string" is used by programmers to indicate a sequence of characters. In practice, you can use regular expressions with whatever data you can access using the application or programming language you are working with. 

## Different Regular Expression Engines

A regular expression "engine" is a piece of software that can process regular expressions, trying to match the pattern to the given string. Usually, the engine is part of a larger application and you do not access the engine directly. Rather, the application will invoke it for you when needed, making sure the right regular expression is applied to the right file or data. As usual in the software world, different regular expression engines are not fully compatible with each other. It is not possible to describe every kind of engine and regular expression syntax (or "flavor") in this tutorial. I will focus on the regex flavor used by Perl 5, for the simple reason that this regex flavor is the most popular one, and deservedly so. Many more recent regex engines are very similar, but not identical, to the one of Perl 5. Examples are the open source PCRE engine (used in many tools and languages like PHP), the .NET 
regular expression library, and the regular expression package included with version 1.4 and later of the Java JDK. I will point out to you whenever differences in regex flavors are important, and which features are specific to the Perl-derivatives mentioned above. 

## Give Regexes A First Try

You can easily try the following yourself in a text editor that supports regular expressions, such as EditPad Pro. If you do not have such an editor, you can download the free evaluation version of EditPad Pro to try this out. EditPad Pro's regex engine is fully functional in the demo version. As a quick test, copy and paste the text of this page into EditPad Pro. Then select Search|Show Search Panel from the menu. In the search pane that appears near the bottom, type in «regex» in the box labeled "Search Text". Mark the "Regular expression" checkbox, and click the Find First button. This is the leftmost button on the search panel. See how EditPad Pro's regex engine finds the first match. Click the Find Next button, which sits next to the Find First button, to find further matches. When there are no further matches, the Find Next button's icon will flash briefly. Now try to search using the regex «reg(ular expressions?|ex(p|es)?)» . This regex will find all names, singular and plural, I have used on this page to say "regex". If we only had plain text search, we would have needed 5 searches. With regexes, we need just one search. Regexes save you time when using a tool like EditPad Pro. Select Count Matches in the Search menu to see how many times this regular expression can match the file you have open in EditPad Pro. 

If you are a programmer, your software will run faster since even a simple regex engine applying the above 

![8_image_0.png](8_image_0.png) Regular expressions also reduce development time. With a regex engine, it takes only one line (e.g. 

in Perl, PHP, Java or .NET) or a couple of lines 
(e.g. in C using PCRE) of code to, say, check if the user's input looks like a valid email address. 

## 2. Literal Characters

The most basic regular expression consists of a single literal character, e.g.: «a». It will match the first occurrence of that character in the string. If the string is "Jack is a boy", it will match the "a" after the 
"J". The fact that this "a" is in the middle of the word does not matter to the regex engine. If it matters to you, you will need to tell that to the regex engine by using word boundaries. We will get to that later. This regex can match the second "a" too. It will only do so when you tell the regex engine to start searching through the string after the first match. In a text editor, you can do so by using its "Find Next" or "Search Forward" function. In a programming language, there is usually a separate function that you can call to continue searching through the string after the previous match. 

Similarly, the regex «cat» will match "cat" in "About cats and dogs". This regular expression consists of a series of three literal characters. This is like saying to the regex engine: find a «c», immediately followed by an «a», immediately followed by a «t». Note that regex engines are case sensitive by default. «cat» does not match "Cat", unless you tell the regex engine to ignore differences in case. 

## Special Characters

Because we want to do more than simply search for literal pieces of text, we need to reserve certain characters for special use. In the regex flavors discussed in this tutorial, there are 11 characters with special meanings: 
the opening square bracket «[», the backslash «\», the caret «^», the dollar sign «$», the period or dot «.», the vertical bar or pipe symbol «|», the question mark «?», the asterisk or star «*», the plus sign «+», the opening round bracket «(» and the closing round bracket «)». These special characters are often called "metacharacters". If you want to use any of these characters as a literal in a regex, you need to escape them with a backslash. If you want to match "1+1=2", the correct regex is «1\+1=2». Otherwise, the plus sign will have a special meaning. Note that «1+1=2», with the backslash omitted, is a valid regex. So you will not get an error message. But it will not match "1+1=2". It would match "111=2" in "123+111=234", due to the special meaning of the plus character. If you forget to escape a special character where its use is not allowed, such as in «+1», then you will get an error message. Most regular expression flavors treat the brace «{» as a literal character, unless it is part of a repetition operator like «{1,3}». So you generally do not need to escape it with a backslash, though you can do so if you want. An exception to this rule is the java.util.regex package: it requires all literal braces to be escaped. All other characters should not be escaped with a backslash. That is because the backslash is also a special character. The backslash in combination with a literal character can create a regex token with a special meaning. E.g. «\d» will match a single digit from 0 to 9. Escaping a single metacharacter with a backslash works in all regular expression flavors. Many flavors also support the \Q...\E escape sequence. All the characters between the \Q and the \E are interpreted as literal characters. E.g. «\Q*\d+*\E» matches the literal text "*\d+*". The \E may be omitted at the end of the regex, so «\Q*\d+*» is the same as «\Q*\d+*\E». This syntax is supported by the JGsoft engine, Perl and PCRE, both inside and outside character classes. Java supports it outside character classes only, and quantifies it as one token. 

## Special Characters And Programming Languages

If you are a programmer, you may be surprised that characters like the single quote and double quote are not special characters. That is correct. When using a regular expression or grep tool like PowerGREP or the search function of a text editor like EditPad Pro, you should not escape or repeat the quote characters like you do in a programming language. In your source code, you have to keep in mind which characters get special treatment inside strings by your programming language. That is because those characters will be processed by the compiler, before the regex library sees the string. So the regex «1\+1=2» must be written as "1\\+1=2" in C++ code. The C++ 
compiler will turn the escaped backslash in the source code into a single backslash in the string that is passed on to the regex library. To match "c:\temp", you need to use the regex «c:\\temp». As a string in C++ source code, this regex becomes "c:\\\\temp". Four backslashes to match a single one indeed. See the tools and languages section in this book for more information on how to use regular expressions in various programming languages. 

## Non-Printable Characters

You can use special character sequences to put non-printable characters in your regular expression. Use «\t» to match a tab character (ASCII 0x09), «\r» for carriage return (0x0D) and «\n» for line feed (0x0A). More exotic non-printables are «\a» (bell, 0x07), «\e» (escape, 0x1B), «\f» (form feed, 0x0C) and «\v» (vertical tab, 0x0B). Remember that Windows text files use "\r\n" to terminate lines, while UNIX text files use "\n". You can include any character in your regular expression if you know its hexadecimal ASCII or ANSI code for the character set that you are working with. In the Latin-1 character set, the copyright symbol is character 0xA9. So to search for the copyright symbol, you can use «\xA9». Another way to search for a tab is to use «\x09». Note that the leading zero is required. Most regex flavors also support the tokens «\cA» through «\cZ» to insert ASCII control characters. The letter after the backslash is always a lowercase c. The second letter is an uppercase letter A through Z, to indicate Control+A through Control+Z. These are equivalent to «\x01» through «\x1A» (26 decimal). E.g. «\cM» matches a carriage return, just like «\r» and «\x0D». In XML Schema regular expressions, «\c» is a shorthand character class that matches any character allowed in an XML name. If your regular expression engine supports Unicode, use «\uFFFF» rather than «\xFF» to insert a Unicode character. The euro currency sign occupies code point 0x20AC. If you cannot type it on your keyboard, you can insert it into a regular expression with «\u20AC». 

# 3. First Look At How A Regex Engine Works Internally

Knowing how the regex engineworks will enable you to craft better regexes more easily. It will help you understand quickly why a particular regex does not do what you initially expected. This will save you lots of guesswork and head scratching when you need to write more complex regexes. 

There are two kinds of regular expression engines: text-directed engines, and regex-directed engines. Jeffrey Friedl calls them DFA and NFA engines, respectively. All the regex flavors treated in this tutorial are based on regex-directed engines. This is because certain very useful features, such as lazy quantifiers and backreferences, can only be implemented in regex-directed engines. No surprise that this kind of engine is more popular. Notable tools that use text-directed engines are awk, egrep, flex, lex, MySQL and Procmail. For awk and egrep, there are a few versions of these tools that use a regex-directed engine. You can easily find out whether the regex flavor you intend to use has a text-directed or regex-directed engine. If backreferences and/or lazy quantifiers are available, you can be certain the engine is regex-directed. You can do the test by applying the regex «regex|regex not» to the string "regex not". If the resulting match is only "regex", the engine is regex-directed. If the result is "regex not", then it is text-directed. The reason behind this is that the regex-directed engine is "eager". 

In this tutorial, after introducing a new regex token, I will explain step by step how the regex engine actually processes that token. This inside look may seem a bit long-winded at certain times. But understanding how the regex engine works will enable you to use its full power and help you avoid common mistakes. 

## The Regex-Directed Engine Always Returns The Leftmost Match

This is a very important point to understand: a regex-directed engine will always return the leftmost match, even if a "better" match could be found later. When applying a regex to a string, the engine will start at the first character of the string. It will try all possible permutations of the regular expression at the first character. 

Only if all possibilities have been tried and found to fail, will the engine continue with the second character in the text. Again, it will try all possible permutations of the regex, in exactly the same order. The result is that the regex-directed engine will return the *leftmost* match. 

When applying «cat» to "He captured a catfish for his cat.", the engine will try to match the first token in the regex «c» to the first character in the match "H". This fails. There are no other possible permutations of this regex, because it merely consists of a sequence of literal characters. So the regex engine tries to match the «c» with the "e". This fails too, as does matching the «c» with the space. Arriving at the 4th character in the match, «c» matches "c". The engine will then try to match the second token «a» to the 5th character, "a". This succeeds too. But then, «t» fails to match "p". At that point, the engine knows the regex cannot be matched starting at the 4th character in the match. So it will continue with the 5th: "a". Again, «c» 
fails to match here and the engine carries on. At the 15th character in the match, «c» again matches "c". The engine then proceeds to attempt to match the remainder of the regex at character 15 and finds that «a» matches "a" and «t» matches "t". The entire regular expression could be matched starting at character 15. The engine is "eager" to report a match. It will therefore report the first three letters of catfish as a valid match. The engine never proceeds beyond this point to see if there are any "better" matches. The first match is considered good enough. In this first example of the engine's internals, our regex engine simply appears to work like a regular text search routine. A text-directed engine would have returned the same result too. However, it is important that you can follow the steps the engine takes in your mind. In following examples, the way the engine works will have a profound impact on the matches it will find. Some of the results may be surprising. But they are always logical and predetermined, once you know how the engine works. 

## 4. Character Classes Or Character Sets

With a "character class", also called "character set", you can tell the regex engine to match only one out of several characters. Simply place the characters you want to match between square brackets. If you want to match an a or an e, use «[ae]». You could use this in «gr[ae]y» to match either "gray" or "grey". Very useful if you do not know whether the document you are searching through is written in American or British English. A character class matches only a single character. «gr[ae]y» will not match "graay", "graey" or any such thing. The order of the characters inside a character class does not matter. The results are identical. You can use a hyphen inside a character class to specify a range of characters. «[0-9]» matches a *single* digit between 0 and 9. You can use more than one range. «[0-9a-fA-F]» matches a single hexadecimal digit, case insensitively. You can combine ranges and single characters. «[0-9a-fxA-FX]» matches a hexadecimal digit or the letter X. Again, the order of the characters and the ranges does not matter. 

## Useful Applications

Find a word, even if it is misspelled, such as «sep[ae]r[ae]te» or «li[cs]en[cs]e». Find an identifier in a programming language with «[A-Za-z_][A-Za-z_0-9]*». 

Find a C-style hexadecimal number with «0[xX][A-Fa-f0-9]+». 

## Negated Character Classes

Typing a caret after the opening square bracket will negate the character class. The result is that the character class will match any character that is not in the character class. Unlike the dot, negated character classes also match (invisible) line break characters. 

It is important to remember that a negated character class still must match a character. «q[^u]» does not mean: "a q not followed by a u". It means: "a q followed by a character that is not a u". It will not match the q in the string "Iraq". It will match the q and the space after the q in "Iraq is a country". Indeed: the space will be part of the overall match, because it is the "character that is not a u" that is matched by the negated character class in the above regexp. If you want the regex to match the q, and only the q, in both strings, you need to use negative lookahead: «q(?!u)». But we will get to that later. 

## Metacharacters Inside Character Classes

Note that the only special characters or metacharacters inside a character class are the closing bracket (]), the backslash (\), the caret (^) and the hyphen (-). The usual metacharacters are normal characters inside a character class, and do not need to be escaped by a backslash. To search for a star or plus, use «[+*]». Your regex will work fine if you escape the regular metacharacters inside a character class, but doing so significantly reduces readability. To include a backslash as a character without any special meaning inside a character class, you have to escape it with another backslash. «[\\x]» matches a backslash or an x. The closing bracket (]), the caret (^) and the hyphen (-) can be included by escaping them with a backslash, or by placing them in a position where they do not take on their special meaning. I recommend the latter method, since it improves readability. To include a caret, place it anywhere except right after the opening bracket. «[x^]» matches an x or a caret. You can put the closing bracket right after the opening bracket, or the negating caret. «[]x]» matches a closing bracket or an x. «[^]x]» matches any character that is not a closing bracket or an x. The hyphen can be included right after the opening bracket, or right before the closing bracket, or right after the negating caret. Both «[-x]» and «[x-]» match an x or a hyphen. You can use all non-printable characters in character classes just like you can use them outside of character classes. E.g. «[$\u20AC]» matches a dollar or euro sign, assuming your regex flavor supports Unicode. The JGsoft engine, Perl and PCRE also support the \Q...\E sequence inside character classes to escape a string of characters. E.g. «[\Q[-]\E]» matches "[", "-" or "]". POSIX regular expressions treat the backslash as a literal character inside character classes. This means you can't use backslashes to escape the closing bracket (]), the caret (^) and the hyphen (-). To use these characters, position them as explained above in this section. This also means that special tokens like shorthands are not available in POSIX regular expressions. See the tutorial topic on POSIX bracket expressions for more information. 

## Shorthand Character Classes

Since certain character classes 

![14_image_0.png](14_image_0.png) are used often, a series of shorthand character classes are available. «\d» is short for «[09]». «\w» stands for "word character". Exactly which characters it matches differs between regex flavors. In all flavors, it will include «[A-Zaz]». In most, the underscore and digits are also included. In some flavors, word characters from other languages may also match. The best way to find out is to do a couple of tests with the regex flavor you are using. In the screen shot, you can see the characters matched by «\w» in RegexBuddy using various scripts. 

«\s» stands for "whitespace character". Again, which characters this actually includes, depends on the regex flavor. In all flavors discussed in this tutorial, it includes «[ \t]». That is: «\s» will match a space or a tab. In most flavors, it also includes a carriage return or a line feed as in «[ \t\r\n]». Some flavors include additional, rarely used non-printable characters such as vertical tab and form feed. Shorthand character classes can be used both inside and outside the square brackets. «\s\d» matches a whitespace character followed by a digit. «[\s\d]» matches a single character that is either whitespace or a digit. When applied to "1 + 2 = 3", the former regex will match " 2" (space two), while the latter matches "1" (one). «[\da-fA-F]» matches a hexadecimal digit, and is equivalent to «[0-9a-fA-F]». 

## Negated Shorthand Character Classes

The above three shorthands also have negated versions. «\D» is the same as «[^\d]», «\W» is short for «[^\w]» and «\S» is the equivalent of «[^\s]». Be careful when using the negated shorthands inside square brackets. «[\D\S]» is not the same as «[^\d\s]». The latter will match any character that is not a digit or whitespace. So it will match "x", but not "8". The former, however, will match any character that is either not a digit, or is not whitespace. Because a digit is not whitespace, and whitespace is not a digit, «[\D\S]» will match any character, digit, whitespace or otherwise. 

## Repeating Character Classes

If you repeat a character class by using the «?», «*» or «+» operators, you will repeat the entire character class, and not just the character that it matched. The regex «[0-9]+» can match "837" as well as "222". If you want to repeat the matched character, rather than the class, you will need to use backreferences. «([09])\1+» will match "222" but not "837". When applied to the string "833337", it will match "3333" in the middle of this string. If you do not want that, you need to use lookahead and lookbehind. But I digress. I did not yet explain how character classes work inside the regex engine. Let us take a look at that first. 

As I already said: the order of the characters inside a character class does not matter. «gr[ae]y» will match "grey" in "Is his hair grey or gray?", because that is the *leftmost match*. We already saw how the engine applies a regex consisting only of literal characters. Below, I will explain how it applies a regex that has more than one permutation. That is: «gr[ae]y» can match both "gray" and "grey". 

Nothing noteworthy happens for the first twelve characters in the string. The engine will fail to match «g» at every step, and continue with the next character in the string. When the engine arrives at the 13th character, "g" is matched. The engine will then try to match the remainder of the regex with the text. The next token in the regex is the literal «r», which matches the next character in the text. So the third token, «[ae]» is attempted at the next character in the text ("e"). The character class gives the engine two options: match «a» 
or match «e». It will first attempt to match «a», and fail. But because we are using a regex-directed engine, it must continue trying to match all the other permutations of the regex pattern before deciding that the regex cannot be matched with the text starting at character 13. So it will continue with the other option, and find that «e» matches "e". The last regex token is «y», which can be matched with the following character as well. The engine has found a complete match with the text starting at character 13. It will return "grey" as the match result, and look no further. Again, the *leftmost match* was returned, even though we put the «a» first in the character class, and "gray" could have been matched in the string. But the engine simply did not get that far, because another equally valid match was found to the left of it. 

# 5. The Dot Matches (Almost) Any Character

In regular expressions, the dot or period is one of the most commonly used metacharacters. Unfortunately, it is also the most commonly misused metacharacter. 

The dot matches a single character, without caring what that character is. The only exception are newlinecharacters. In all regex flavors discussed in this tutorial, the dot will not match a newline character by default. So by default, the dot is short for the negated character class «[^\n]» (UNIX regex flavors) or «[^\r\n]» (Windows regex flavors). This exception exists mostly because of historic reasons. The first tools that used regular expressions were line-based. They would read a file line by line, and apply the regular expression separately to each line. The effect is that with these tools, the string could never contain newlines, so the dot could never match them. 

Modern tools and languages can apply regular expressions to very large strings or even entire files. All regex flavors discussed here have an option to make the dot match all characters, including newlines. In RegexBuddy, EditPad Pro or PowerGREP, you simply tick the checkbox labeled "dot matches newline". In Perl, the mode where the dot also matches newlines is called "single-line mode". This is a bit unfortunate, because it is easy to mix up this term with "multi-line mode". Multi-line mode only affects anchors, and single-line mode only affects the dot. You can activate single-line mode by adding an s after the regex code, like this: m/^regex$/s;. 

Other languages and regex libraries have adopted Perl's terminology. When using the regex classes of the 
.NET framework, you activate this mode by specifying RegexOptions.Singleline, such as in Regex.Match("string", "regex", RegexOptions.Singleline). In all programming languages and regex libraries I know, activating single-line mode has no effect other than making the dot match newlines. So if you expose this option to your users, please give it a clearer label like was done in RegexBuddy, EditPad Pro and PowerGREP. 

JavaScript and VBScript do not have an option to make the dot match line break characters. In those languages, you can use a character class such as «[\s\S]» to match any character. This character matches a character that is either a whitespace character (including line break characters), or a character that is not a whitespace character. Since all characters are either whitespace or non-whitespace, this character class matches any character. 

## Use The Dot Sparingly

The dot is a very powerful regex metacharacter. It allows you to be lazy. Put in a dot, and everything will match just fine when you test the regex on valid data. The problem is that the regex will also match in cases where it should not match. If you are new to regular expressions, some of these cases may not be so obvious at first. I will illustrate this with a simple example. Let's say we want to match a date in mm/dd/yy format, but we want to leave the user the choice of date separators. The quick solution is «\d\d.\d\d.\d\d». Seems fine at first. It will match a date like "02/12/03" just fine. Trouble is: "02512703" is also considered a valid date by this regular expression. In this match, the first dot matched "5", and the second matched "7". Obviously not what we intended. «\d\d[- /.]\d\d[- /.]\d\d» is a better solution. This regex allows a dash, space, dot and forward slash as date separators. Remember that the dot is not a metacharacter inside a character class, so we do not need to escape it with a backslash. 

This regex is still far from perfect. It matches "99/99/99" as a valid date. «[0-1]\d[- /.][0-3]\d[- 
/.]\d\d» is a step ahead, though it will still match "19/39/99". How perfect you want your regex to be depends on what you want to do with it. If you are validating user input, it has to be perfect. If you are parsing data files from a known source that generates its files in the same way every time, our last attempt is probably more than sufficient to parse the data without errors. You can find a better regex to match dates in the example section. 

## Use Negated Character Sets Instead Of The Dot

I will explain this in depth when I present you the repeat operators star and plus, but the warning is important enough to mention it here as well. I will illustrate with an example. Suppose you want to match a double-quoted string. Sounds easy. We can have any number of any character between the double quotes, so «".*"» seems to do the trick just fine. The dot matches any character, and the star allows the dot to be repeated any number of times, including zero. If you test this regex on "Put a 
"string" between double quotes", it will match ""string"" just fine. Now go ahead and test it on 
"Houston, we have a problem with "string one" and "string two". Please respond." Ouch. The regex matches ""string one" and "string two"". Definitely not what we intended. The reason for this is that the star is *greedy*. In the date-matching example, we improved our regex by replacing the dot with a character class. Here, we will do the same. Our original definition of a double-quoted string was faulty. We do not want any number of any character between the quotes. We want any number of characters that are not double quotes or newlines between the quotes. So the proper regex is «"[^"\r\n]*"». 

## 6. Start Of String And End Of String Anchors

Thus far, I have explained literal characters and character classes. In both cases, putting one in a regex will cause the regex engine to try to match a single character. 

Anchors are a different breed. They do not match any character at all. Instead, they match a position before, after or between characters. They can be used to "anchor" the regex match at a certain position. The caret «^» 
matches the position before the first character in the string. Applying «^a» to "abc" matches "a". «^b» will not match "abc" at all, because the «b» cannot be matched right after the start of the string, matched by «^». See below for the inside view of the regex engine. Similarly, «$» matches right after the last character in the string. «c$» matches "c" in "abc", while «a$» does not match at all. 

## Useful Applications

When using regular expressions in a programming language to validate user input, using anchors is very important. If you use the code if ($input =~ m/\d+/) in a Perl script to see if the user entered an integer number, it will accept the input even if the user entered "qsdf4ghjk", because «\d+» matches the 4. The correct regex to use is «^\d+$». Because "start of string" must be matched before the match of «\d+», and "end of string" must be matched right after it, the entire string must consist of digits for «^\d+$» to be able to match. 

It is easy for the user to accidentally type in a space. When Perl reads from a line from a text file, the line break will also be stored in the variable. So before validating input, it is good practice to trim leading and trailing whitespace. «^\s+» matches leading whitespace and «\s+$» matches trailing whitespace. In Perl, you could use $input =~ s/^\s+|\s+$//g. Handy use of alternation and /g allows us to do this in a single line of code. 

## Using ^ And $ As Start Of Line And End Of Line Anchors

If you have a string consisting of multiple lines, like "first line\nsecond line" (where \n indicates a line break), it is often desirable to work with lines, rather than the entire string. Therefore, all the regex engines discussed in this tutorial have the option to expand the meaning of both anchors. «^» can then match at the start of the string (before the "f" in the above string), as well as after each line break (between "\n" and "s"). Likewise, «$» will still match at the end of the string (after the last "e"), and also before every line break (between "e" and "\n"). 

In text editors like EditPad Pro or GNU Emacs, and regex tools like PowerGREP, the caret and dollar always match at the start and end of each line. This makes sense because those applications are designed to work with entire files, rather than short strings. In all programming languages and libraries discussed in this book , except Ruby, you have to explicitly activate this extended functionality. It is traditionally called "multi-line mode". In Perl, you do this by adding an m after the regex code, like this: m/^regex$/m;. In .NET, the anchors match before and after newlines when you specify RegexOptions.Multiline, such as in Regex.Match("string", "regex", 
RegexOptions.Multiline). 

## Permanent Start Of String And End Of String Anchors

«\A» only ever matches at the start of the string. Likewise, «\Z» only ever matches at the end of the string. 

These two tokens never match at line breaks. This is true in all regex flavors discussed in this tutorial, even when you turn on "multiline mode". In EditPad Pro and PowerGREP, where the caret and dollar always match at the start and end of lines, «\A» and «\Z» only match at the start and the end of the entire file. 

## Zero-Length Matches

We saw that the anchors match at a position, rather than matching a character. This means that when a regex only consists of one or more anchors, it can result in a zero-length match. Depending on the situation, this can be very useful or undesirable. Using «^\d*$» to test if the user entered a number (notice the use of the star instead of the plus), would cause the script to accept an empty string as a valid input. See below. 

However, matching only a position can be very useful. In email, for example, it is common to prepend a 
"greater than" symbol and a space to each line of the quoted message. In VB.NET, we can easily do this with Dim Quoted as String = Regex.Replace(Original, "^", "> ", RegexOptions.Multiline). We are using multi-line mode, so the regex «^» matches at the start of the quoted message, and after each newline. The Regex.Replace method will remove the regex match from the string, and insert the replacement string (greater than symbol and a space). Since the match does not include any characters, nothing is deleted. 

However, the match does include a starting position, and the replacement string is inserted there, just like we want it. 

## Strings Ending With A Line Break

Even though «\Z» and «$» only match at the end of the string (when the option for the caret and dollar to match at embedded line breaks is off), there is one exception. If the string ends with a line break, then «\Z» and «$» will match at the position before that line break, rather than at the very end of the string. This 
"enhancement" was introduced by Perl, and is copied by many regex flavors, including Java, .NET and PCRE. In Perl, when reading a line from a file, the resulting string will end with a line break. Reading a line from a file with the text "joe" results in the string "joe\n". When applied to this string, both «^[a-z]+$» 
and «\A[a-z]+\Z» will match "joe". 

If you only want a match at the absolute very end of the string, use «\z» (lower case z instead of upper case Z). «\A[a-z]+\z» does not match "joe\n". «\z» matches after the line break, which is not matched by the character class. 

Let's see what happens when we try to match «^4$» to "749\n486\n4" (where \n represents a newline character) in multi-line mode. As usual, the regex engine starts at the first character: "7". The first token in the regular expression is «^». Since this token is a zero-width token, the engine does not try to match it with the character, but rather with the position before the character that the regex engine has reached so far. «^» 
indeed matches the position before "7". The engine then advances to the next regex token: «4». Since the previous token was zero-width, the regex engine does not advance to the next character in the string. It remains at "7". «4» is a literal character, which does not match "7". There are no other permutations of the regex, so the engine starts again with the first regex token, at the next character: "4". This time, «^» cannot match at the position before the 4. This position is preceded by a character, and that character is not a newline. The engine continues at "9", and fails again. The next attempt, at "\n", also fails. Again, the position before "\n" is preceded by a character, "9", and that character is not a newline. Then, the regex engine arrives at the second "4" in the string. The «^» can match at the position before the "4", because it is preceded by a newline character. Again, the regex engine advances to the next regex token, «4», but does not advance the character position in the string. «4» matches "4", and the engine advances both the regex token and the string character. Now the engine attempts to match «$» at the position before 
(indeed: before) the "8". The dollar cannot match here, because this position is followed by a character, and that character is not a newline. Yet again, the engine must try to match the first token again. Previously, it was successfully matched at the second "4", so the engine continues at the next character, "8", where the caret does not match. Same at the six and the newline. Finally, the regex engine tries to match the first token at the third "4" in the string. With success. After that, the engine successfully matches «4» with "4". The current regex token is advanced to «$», and the current character is advanced to the very last position in the string: the void after the string. No regex token that needs a character to match can match here. Not even a negated character class. However, we are trying to match a dollar sign, and the mighty dollar is a strange beast. It is zero-width, so it will try to match the position before the current character. It does not matter that this "character" is the void after the string. In fact, the dollar will check the current character. It must be either a newline, or the void after the string, for «$» to match the position before the current character. Since that is the case after the example, the dollar matches successfully. Since «$» was the last token in the regex, the engine has found a successful match: the last "4" in the string. 

## Another Inside Look

Earlier I mentioned that «^\d*$» would successfully match an empty string. Let's see why. There is only one 
"character" position in an empty string: the void after the string. The first token in the regex is «^». It matches the position before the void after the string, because it is preceded by the void before the string. The next token is «\d*». As we will see later, one of the star's effects is that it makes the «\d», in this case, optional. The engine will try to match «\d» with the void after the string. That fails, but the star turns the failure of the 
«\d» into a zero-width success. The engine will proceed with the next regex token, without advancing the position in the string. So the engine arrives at «$», and the void after the string. We already saw that those match. At this point, the entire regex has matched the empty string, and the engine reports success. 

## Caution For Programmers

A regular expression such as «$» all by itself can indeed match after the string. If you would query the engine for the character position, it would return the length of the string if string indices are zero-based, or the length+1 if string indices are one-based in your programming language. If you would query the engine for the length of the match, it would return zero. What you have to watch out for is that String[Regex.MatchPosition] may cause an access violation or segmentation fault, because MatchPosition can point to the void after the string. This can also happen with «^» and «^$» if the last character in the string is a newline. 

## 7. Word Boundaries

The metacharacter «\b» is an anchor like the caret and the dollar sign. It matches at a position that is called a 
"word boundary". This match is zero-length. 

There are four different positions that qualify as word boundaries: 
- Before the first character in the string, if the first character is a word character. - After the last character in the string, if the last character is a word character. - Between a word character and a non-word character following right after the word character. - Between a non-word character and a word character following right after the non-word character. 

Simply put: «\b» allows you to perform a "whole words only" search using a regular expression in the form of «\bword\b». A "word character" is a character that can be used to form words. All characters that are not "word characters" are "non-word characters". The exact list of characters is different for each regex flavor, but all word characters are always matched by the short-hand character class «\w». All non-word characters are always matched by «\W». 

In Perl and the other regex flavors discussed in this tutorial, there is only one metacharacter that matches both before a word and after a word. This is because any position between characters can never be both at the start and at the end of a word. Using only one operator makes things easier for you. Note that «\w» usually also matches digits. So «\b4\b» can be used to match a 4 that is not part of a larger number. This regex will not match "44 sheets of a4". So saying "«\b» matches before and after an alphanumeric sequence" is more exact than saying "before and after a word". 

## Negated Word Boundary

«\B» is the negated version of «\b». «\B» matches at every position where «\b» does not. Effectively, «\B» matches at any position between two word characters as well as at any position between two non-word characters. 

Let's see what happens when we apply the regex «\bis\b» to the string "This island is beautiful". The engine starts with the first token «\b» at the first character "T". Since this token is zero-length, the position before the character is inspected. «\b» matches here, because the T is a word character and the character before it is the void before the start of the string. The engine continues with the next token: the literal «i». The engine does not advance to the next character in the string, because the previous regex token was zero-width. «i» does not match "T", so the engine retries the first token at the next character position. 

«\b» cannot match at the position between the "T" and the "h". It cannot match between the "h" and the 
"i" either, and neither between the "i" and the "s". The next character in the string is a space. «\b» matches here because the space is not a word character, and the preceding character is. Again, the engine continues with the «i» which does not match with the space. Advancing a character and restarting with the first regex token, «\b» matches between the space and the second "i" in the string. Continuing, the regex engine finds that «i» matches "i" and «s» matches "s". Now, the engine tries to match the second «\b» at the position before the "l". This fails because this position is between two word characters. The engine reverts to the start of the regex and advances one character to the 
"s" in "island". Again, the «\b» fails to match and continues to do so until the second space is reached. It matches there, but matching the «i» fails. But «\b» matches at the position before the third "i" in the string. The engine continues, and finds that «i» 
matches "i" and «s» matches «s». The last token in the regex, «\b», also matches at the position before the second space in the string because the space is not a word character, and the character before it is. 

The engine has successfully matched the word "is" in our string, skipping the two earlier occurrences of the characters i and s. If we had used the regular expression «is», it would have matched the "is" in "This". 

## Tcl Word Boundaries

Word boundaries, as described above, are supported by all regular expression flavors described in in this book , except for the two POSIX RE flavors and the Tcl regexp command. POSIX does not support word boundaries at all. Tcl uses a different syntax. In Tcl, «\b» matches a backspace character, just like «\x08» in most regex flavors (including Tcl's). «\B» matches a single backslash character in Tcl, just like «\\» in all other regex flavors (and Tcl too). Tcl uses the letter "y" instead of the letter "b" to match word boundaries. «\y» matches at any word boundary position, while «\Y» matches at any position that is not a word boundary. These Tcl regex tokens match exactly the same as «\b» and «\B» in Perl-style regex flavors. They don't discriminate between the start and the end of a word. Tcl has two more word boundary tokens that do discriminate between the start and end of a word. «\m» matches only at the start of a word. That is, it matches at any position that has a non-word character to the left of it, and a word character to the right of it. It also matches at the start of the string if the first character in the string is a word character. «\M» matches only at the end of a word. It matches at any position that has a word character to the left of it, and a non-word character to the right of it. It also matches at the end of the string if the last character in the string is a word character. The only regex engine that supports Tcl-style word boundaries (besides Tcl itself) is the JGsoft engine. In PowerGREP and EditPad Pro, «\b» and «\B» are Perl-style word boundaries, and «\y», «\Y», «\m» and «\M» 
are Tcl-style word boundaries. In most situations, the lack of «\m» and «\M» tokens is not a problem. «\yword\y» finds "whole words only" occurrences of "word" just like «\mword\M» would. «\Mword\m» could never match anywhere, since «\M» never matches at a position followed by a word character, and «\m» never at a position preceded by one. If your regular expression needs to match characters before or after «\y», you can easily specify in the regex whether these characters should be word characters or non-word characters. E.g. if you want to match any word, «\y\w+\y» will give the same result as «\m.+\M». Using «\w» instead of the dot automatically restricts the first «\y» to the start of a word, and the second «\y» to the end of a word. Note that «\y.+\y» would not work. This regex matches each word, and also each sequence of non-word characters between the words in your subject string. That said, if your flavor supports «\m» and «\M», the regex engine could apply «\m\w+\M» 
slightly faster than «\y\w+\y», depending on its internal optimizations. If your regex flavor supports lookahead and lookbehind, you can use «(?<!\w)(?=\w)» to emulate Tcl's «\m» and «(?<=\w)(?!\w)» to emulate «\M». Though quite a bit more verbose, these lookaround constructs match exactly the same as Tcl's word boundaries. If your flavor has lookahead but not lookbehind, and also has Perl-style word boundaries, you can use 
«\b(?=\w)» to emulate Tcl's «\m» and «\b(?!\w)» to emulate «\M». «\b» matches at the start or end of a word, and the lookahead checks if the next character is part of a word or not. If it is we're at the start of a word. Otherwise, we're at the end of a word. 

# 8. Alternation With The Vertical Bar Or Pipe Symbol

I already explained how you can use character classes to match a single character out of several possible characters. Alternation is similar. You can use alternation to match a single regular expression out of several possible regular expressions. 

If you want to search for the literal text «cat» or «dog», separate both options with a vertical bar or pipe symbol: «cat|dog». If you want more options, simply expand the list: «cat|dog|mouse|fish» . The alternation operator has the lowest precedence of all regex operators. That is, it tells the regex engine to match either everything to the left of the vertical bar, or everything to the right of the vertical bar. If you want to limit the reach of the alternation, you will need to use round brackets for grouping. If we want to improve the first example to match whole words only, we would need to use «\b(cat|dog)\b». This tells the regex engine to find a word boundary, then either "cat" or "dog", and then another word boundary. If we had omitted the round brackets, the regex engine would have searched for "a word boundary followed by cat", or, "dog followed by a word boundary. 

## Remember That The Regex Engine Is Eager

I already explained that the regex engine is eager. It will stop searching as soon as it finds a valid match. The consequence is that in certain situations, the order of the alternatives matters. Suppose you want to use a regex to match a list of function names in a programming language: Get, GetValue, Set or SetValue. The obvious solution is «Get|GetValue|Set|SetValue». Let's see how this works out when the string is "SetValue". The regex engine starts at the first token in the regex, «G», and at the first character in the string, "S". The match fails. However, the regex engine studied the entire regular expression before starting. So it knows that this regular expression uses alternation, and that the entire regex has not failed yet. So it continues with the second option, being the second «G» in the regex. The match fails again. The next token is the first «S» in the regex. The match succeeds, and the engine continues with the next character in the string, as well as the next token in the regex. The next token in the regex is the «e» after the «S» that just successfully matched. «e» 
matches "e". The next token, «t» matches "t". 

At this point, the third option in the alternation has been successfully matched. Because the regex engine is eager, it considers the entire alternation to have been successfully matched as soon as one of the options has. 

In this example, there are no other tokens in the regex outside the alternation, so the entire regex has successfully matched "Set" in "SetValue". Contrary to what we intended, the regex did not match the entire string. There are several solutions. One option is to take into account that the regex engine is eager, and change the order of the options. If we use 
«GetValue|Get|SetValue|Set», «SetValue» will be attempted before «Set», and the engine will match the entire string. We could also combine the four options into two and use the question mark to make part of them optional: «Get(Value)?|Set(Value)?». Because the question mark is greedy, «SetValue» will be attempted before «Set». The best option is probably to express the fact that we only want to match complete words. We do not want to match Set or SetValue if the string is "SetValueFunction". So the solution is 
«\b(Get|GetValue|Set|SetValue)\b» or «\b(Get(Value)?|Set(Value)?)\b». Since all options have the same end, we can optimize this further to «\b(Get|Set)(Value)?\b» . All regex flavors discussed in this book work this way, except one: the POSIX standard mandates that the longest match be returned, regardless if the regex engine is implemented using an NFA or DFA algorithm. 

## 9. Optional Items

The question mark makes the preceding token in the regular expression optional. E.g.: «colou?r» matches both "colour" and "color". 

You can make several tokens optional by grouping them together using round brackets, and placing the question mark after the closing bracket. E.g.: «Nov(ember)?» will match "Nov" and "November". You can write a regular expression that matches many alternatives by including more than one question mark. 

«Feb(ruary)? 23(rd)?» matches "February 23rd", "February 23", "Feb 23rd" and "Feb 23". 

## Important Regex Concept: Greediness

With the question mark, I have introduced the first metacharacter that is *greedy*. The question mark gives the regex engine two choices: try to match the part the question mark applies to, or do not try to match it. The engine will always try to match that part. Only if this causes the entire regular expression to fail, will the engine try ignoring the part the question mark applies to. The effect is that if you apply the regex «Feb 23(rd)?» to the string "Today is Feb 23rd, 2003", the match will always be "Feb 23rd" and not "Feb 23". You can make the question mark *lazy* (i.e. turn off the greediness) by putting a second question mark after the first. 

I will say a lot more about greediness when discussing the other repetition operators. 

Let's apply the regular expression «colou?r» to the string "The colonel likes the color green". The first token in the regex is the literal «c». The first position where it matches successfully is the "c" in 
"colonel". The engine continues, and finds that «o» matches "o", «l» matches "l" and another «o» matches 
"o". Then the engine checks whether «u» matches "n". This fails. However, the question mark tells the regex engine that failing to match «u» is acceptable. Therefore, the engine will skip ahead to the next regex token: «r». But this fails to match "n" as well. Now, the engine can only conclude that the entire regular expression cannot be matched starting at the "c" in "colonel". Therefore, the engine starts again trying to match «c» to the first o in "colonel". After a series of failures, «c» will match with the "c" in "color", and «o», «l» and «o» match the following characters. Now the engine checks whether «u» matches "r". This fails. Again: no problem. The question mark allows the engine to continue with «r». This matches "r" and the engine reports that the regex successfully matched "color" in our string. 

## 10. Repetition With Star And Plus

I already introduced one repetition operator or quantifier: the question mark. It tells the engine to attempt match the preceding token zero times or once, in effect making it optional. 

The asterisk or star tells the engine to attempt to match the preceding token zero or more times. The plus tells the engine to attempt to match the preceding token once or more. «<[A-Za-z][A-Za-z0-9]*>» 
matches an HTML tag without any attributes. The sharp brackets are literals. The first character class matches a letter. The second character class matches a letter or digit. The star repeats the second character class. Because we used the star, it's OK if the second character class matches nothing. So our regex will match a tag like "<B>". When matching "<HTML>", the first character class will match "H". The star will cause the second character class to be repeated three times, matching "T", "M" and "L" with each step. I could also have used «<[A-Za-z0-9]+>». I did not, because this regex would match "<1>", which is not a valid HTML tag. But this regex may be sufficient if you know the string you are searching through does not contain any such invalid tags. 

## Limiting Repetition

Modern regex flavors, like those discussed in this tutorial, have an additional repetition operator that allows you to specify how many times a token can be repeated. The syntax is {min,max}, where min is a positive integer number indicating the minimum number of matches, and max is an integer equal to or greater than min indicating the maximum number of matches. If the comma is present but max is omitted, the maximum number of matches is infinite. So «{0,}» is the same as «*», and «{1,}» is the same as «+». Omitting both the comma and max tells the engine to repeat the token exactly min times. You could use «\b[1-9][0-9]{3}\b» to match a number between 1000 and 9999. «\b[1-9][09]{2,4}\b» matches a number between 100 and 99999. Notice the use of the word boundaries. 

## Watch Out For The Greediness!

Suppose you want to use a regex to match an HTML tag. You know that the input will be a valid HTML file, so the regular expression does not need to exclude any invalid use of sharp brackets. If it sits between sharp brackets, it is an HTML tag. 

Most people new to regular expressions will attempt to use «<.+>». They will be surprised when they test it on a string like "This is a <EM>first</EM> test". You might expect the regex to match "<EM>" and when continuing after that match, "</EM>". But it does not. The regex will match "<EM>first</EM>". Obviously not what we wanted. The reason is that the plus is *greedy*. That is, the plus causes the regex engine to repeat the preceding token as often as possible. Only if that causes the entire regex to fail, will the regex engine *backtrack*. That is, it will go back to the plus, make it give up the last iteration, and proceed with the remainder of the regex. Let's take a look inside the regex engine to see in detail how this works and why this causes our regex to fail. After that, I will present you with two possible solutions. Like the plus, the star and the repetition using curly braces are greedy. 

The first token in the regex is «<». This is a literal. As we already know, the first place where it will match is the first "<" in the string. The next token is the dot, which matches any character except newlines. The dot is repeated by the plus. The plus is *greedy*. Therefore, the engine will repeat the dot as many times as it can. The dot matches "E", so the regex continues to try to match the dot with the next character. "M" is matched, and the dot is repeated once more. The next character is the ">". You should see the problem by now. The dot matches the ">", and the engine continues repeating the dot. The dot will match all remaining characters in the string. The dot fails when the engine has reached the void after the end of the string. Only at this point does the regex engine continue with the next token: «>». So far, «<.+» has matched "<EM>first</EM> test" and the engine has arrived at the end of the string. «>» cannot match here. The engine remembers that the plus has repeated the dot more often than is required. (Remember that the plus *requires* the dot to match only once.) Rather than admitting failure, the engine will backtrack. It will reduce the repetition of the plus by one, and then continue trying the remainder of the regex. 

So the match of «.+» is reduced to "EM>first</EM> tes". The next token in the regex is still «>». But now the next character in the string is the last "t". Again, these cannot match, causing the engine to backtrack further. The total match so far is reduced to "<EM>first</EM> te". But «>» still cannot match. So the engine continues backtracking until the match of «.+» is reduced to "EM>first</EM". Now, «>» can match the next character in the string. The last token in the regex has been matched. The engine reports that 
"<EM>first</EM>" has been successfully matched. Remember that the regex engine is *eager* to return a match. It will not continue backtracking further to see if there is another possible match. It will report the first valid match it finds. Because of greediness, this is the leftmost longest match. 

## Laziness Instead Of Greediness

The quick fix to this problem is to make the plus *lazy* instead of greedy. Lazy quantifiers are sometimes also called "ungreedy" or "reluctant". You can do that by putting a question markbehind the plus in the regex. 

You can do the same with the star, the curly braces and the question mark itself. So our example becomes 
«<.+?>». Let's have another look inside the regex engine. 

Again, «<» matches the first "<" in the string. The next token is the dot, this time repeated by a lazy plus. This tells the regex engine to repeat the dot as few times as possible. The minimum is one. So the engine matches the dot with "E". The requirement has been met, and the engine continues with «>» and "M". This fails. Again, the engine will *backtrack*. But this time, the backtracking will force the lazy plus to expand rather than reduce its reach. So the match of «.+» is expanded to "EM", and the engine tries again to continue with «>». 

Now, ">" is matched successfully. The last token in the regex has been matched. The engine reports that "<EM>" has been successfully matched. That's more like it. 

## An Alternative To Laziness

In this case, there is a better option than making the plus lazy. We can use a greedy plus and a negated character class: «<[^>]+>». The reason why this is better is because of the backtracking. When using the lazy plus, the engine has to backtrack for each character in the HTML tag that it is trying to match. When using the negated character class, no backtracking occurs at all when the string contains valid HTML code. Backtracking slows down the regex engine. You will not notice the difference when doing a single search in a text editor. But you will save plenty of CPU cycles when using such a regex is used repeatedly in a tight loop in a script that you are writing, or perhaps in a custom syntax coloring scheme for EditPad Pro. Finally, remember that this tutorial only talks about regex-directed engines. Text-directed engines do not backtrack. They do not get the speed penalty, but they also do not support lazy repetition operators. 

## Repeating \Q...\E Escape Sequences

The \Q...\E sequence escapes a string of characters, matching them as literal characters. The JGsoft engine, Perl and PCRE treat the escaped characters as individual characters. If you place a quantifier after the \E, it will only be applied to the last character. E.g. if you apply «\Q*\d+*\E+» to "*\d+**\d+*", the match will be "*\d+**". Only the asterisk is repeated. (The plus repeats a token one or more times, as I'll explain later in this tutorial.) The Java engine, however, applies the quantifier to the whole \Q...\E sequence. So in Java, the above example matches the whole subject string "*\d+**\d+*". If you want Java to return the same match as Perl, you'll need to split off the asterisk from the escape sequence, like this: «\Q*\d+\E\*+». If you want Perl to repeat the whole sequence like Java does, simply group it: «(?:\Q*\d+*\E)+». 

## 11. Use Round Brackets For Grouping

By placing part of a regular expression inside round brackets or parentheses, you can group that part of the regular expression together. This allows you to apply a regex operator, e.g. a repetition operator, to the entire group. I have already used round brackets for this purpose in previous topics throughout this tutorial. 

Note that only round brackets can be used for grouping. Square brackets define a character class, and curly braces are used by a special repetition operator. 

## Round Brackets Create A Backreference

Besides grouping part of a regular expression together, round brackets also create a "backreference". A 
backreference stores the part of the string matched by the part of the regular expression inside the parentheses. That is, unless you use non-capturing parentheses. Remembering part of the regex match in a backreference, slows down the regex engine because it has more work to do. If you do not use the backreference, you can speed things up by using non-capturing parentheses, at the expense of making your regular expression slightly harder to read. The regex «Set(Value)?» matches "Set" or "SetValue". In the first case, the first backreference will be empty, because it did not match anything. In the second case, the first backreference will contain "Value". If you do not use the backreference, you can optimize this regular expression into «Set(?:Value)?». The question mark and the colon after the opening round bracket are the special syntax that you can use to tell the regex engine that this pair of brackets should not create a backreference. Note the question mark after the opening bracket is unrelated to the question mark at the end of the regex. That question mark is the regex operator that makes the previous token optional. This operator cannot appear after an opening round bracket, because an opening bracket by itself is not a valid regex token. Therefore, there is no confusion between the question mark as an operator to make a token optional, and the question mark as a character to change the properties of a pair of round brackets. The colon indicates that the change we want to make is to turn off capturing the backreference. 

## How To Use Backreferences

Backreferences allow you to reuse part of the regex match. You can reuse it inside the regular expression (see below), or afterwards. What you can do with it afterwards, depends on the tool you are using. In EditPad Pro or PowerGREP, you can use the backreference in the replacement text during a search-and-replace operation by typing \1 (backslash one) into the replacement text. If you searched for «EditPad (Lite|Pro)» and use 
"\1 version" as the replacement, the actual replacement will be "Lite version" in case "EditPad Lite" was matched, and "Pro version" in case "EditPad Pro" was matched. EditPad Pro and PowerGREP have a unique feature that allows you to change the case of the backreference. \U1 inserts the first backreference in uppercase, \L1 in lowercase and \F1 with the first character in uppercase and the remainder in lowercase. Finally, \I1 inserts it with the first letter of each word capitalized, and the other letters in lowercase. 

Regex libraries in programming languages also provide access to the backreference. In Perl, you can use the magic variables $1, $2, etc. to access the part of the string matched by the backreference. In .NET (dot net), 
you can use the Match object that is returned by the Match method of the Regex class. This object has a property called Groups, which is a collection of Group objects. To get the string matched by the third backreference in C\#, you can use MyMatch.Groups[3].Value. 

The .NET (dot net) Regex class also has a method Replace that can do a regex-based search-and-replace on a string. In the replacement text, you can use $1, $2, etc. to insert backreferences. To figure out the number of a particular backreference, scan the regular expression from left to right and count the opening round brackets. The first bracket starts backreference number one, the second number two, etc. Non-capturing parentheses are not counted. This fact means that non-capturing parentheses have another benefit: you can insert them into a regular expression without changing the numbers assigned to the backreferences. This can be very useful when modifying a complex regular expression. 

## The Entire Regex Match As Backreference Zero

Certain tools make the entire regex match available as backreference zero. In EditPad Pro or PowerGREP, you can use the entire regex match in the replacement text during a search and replace operation by typing \0 (backslash zero) into the replacement text. In Perl, the magic variable $& holds the entire regex match. 

Libraries like .NET (dot net) where backreferences are made available as an array or numbered list, the item with index zero holds the entire regex match. Using backreference zero is more efficient than putting an extra pair of round brackets around the entire regex, because that would force the engine to continuously keep an extra copy of the entire regex match. 

## Using Backreferences In The Regular Expression

Backreferences can not only be used after a match has been found, but also during the match. Suppose you want to match a pair of opening and closing HTML tags, and the text in between. By putting the opening tag into a backreference, we can reuse the name of the tag for the closing tag. Here's how: «<([A-Z][A-Z09]*)[^>]*>.*?</\1>» . This regex contains only one pair of parentheses, which capture the string matched by «[A-Z][A-Z0-9]*» into the first backreference. This backreference is reused with «\1» (backslash one). 

The «/» before it is simply the forward slash in the closing HTML tag that we are trying to match. You can reuse the same backreference more than once. «([a-c])x\1x\1» will match "axaxa", "bxbxb" and "cxcxc". If a backreference was not used in a particular match attempt (such as in the first example where the question mark made the first backreference optional), it is simply empty. Using an empty backreference in the regex is perfectly fine. It will simply be replaced with nothingness. A backreference cannot be used inside itself. «([abc]\1)» will not work. Depending on your regex flavor, it will either give an error message, or it will fail to match anything without an error message. Therefore, \0 cannot be used inside a regex, only in the replacement. 

Let's see how the regex engine applies the above regex to the string "Testing <B><I>bold italic</I></B> text". The first token in the regex is the literal «<». The regex engine will traverse the string until it can match at the first "<" in the string. The next token is «[A-Z]». The regex engine also takes note that it is now inside the first pair of capturing parentheses. «[A-Z]» matches "B". The engine advances to «[A-Z0-9]» and ">". This match fails. However, because of the star, that's perfectly fine. The position in the string remains at ">". The position in the regex is advanced to «[^>]». 

This step crosses the closing bracket of the first pair of capturing parentheses. This prompts the regex engine to store what was matched inside them into the first backreference. In this case, "B" is stored. After storing the backreference, the engine proceeds with the match attempt. «[^>]» does not match ">". 

Again, because of another star, this is not a problem. The position in the string remains at ">", and position in the regex is advanced to «>». These obviously match. The next token is a dot, repeated by a lazy star. Because of the laziness, the regex engine will initially skip this token, taking note that it should backtrack in case the remainder of the regex fails. 

The engine has now arrived at the second «<» in the regex, and the second "<" in the string. These match. 

The next token is «/». This does not match "I", and the engine is forced to backtrack to the dot. The dot matches the second "<" in the string. The star is still lazy, so the engine again takes note of the available backtracking position and advances to «<» and "I". These do not match, so the engine again backtracks. The backtracking continues until the dot has consumed "<I>bold italic". At this point, «<» matches the third "<" in the string, and the next token is «/» which matches "/". The next token is «\1». Note that the token the backreference, and not «B». The engine does not substitute the backreference in the regular expression. Every time the engine arrives at the backreference, it will read the value that was stored. This means that if the engine had backtracked beyond the first pair of capturing parentheses before arriving the second time at «\1», the new value stored in the first backreference would be used. But this did not happen here, so "B" it is. This fails to match at "I", so the engine backtracks again, and the dot consumes the third 
"<" in the string. Backtracking continues again until the dot has consumed "<I>bold italic</I>". At this point, «<» matches "<" and «/» matches "/". The engine arrives again at «\1». The backreference still holds "B". «B» matches "B". The last token in the regex, «>» matches ">". A complete match has been found: "<B><I>bold italic</I></B>". 

## Repetition And Backreferences

As I mentioned in the above inside look, the regex engine does not permanently substitute backreferences in the regular expression. It will use the last match saved into the backreference each time it needs to be used. If a new match is found by capturing parentheses, the previously saved match is overwritten. There is a clear difference between «([abc]+)» and «([abc])+». Though both successfully match "cab", the first regex will put "cab" into the first backreference, while the second regex will only store "b". That is because in the second regex, the plus caused the pair of parentheses to repeat three times. The first time, "c" was stored. 

The second time "a" and the third time "b". Each time, the previous value was overwritten, so "b" remains. This also means that «([abc]+)=\1» will match "cab=cab", and that «([abc])+=\1» will not. The reason is that when the engine arrives at «\1», it holds «b» which fails to match "c". Obvious when you look at a simple example like this one, but a common cause of difficulty with regular expressions nonetheless. When using backreferences, always double check that you are really capturing what you want. 

## Useful Example: Checking For Doubled Words

When editing text, doubled words such as "the the" easily creep in. Using the regex «\b(\w+)\s+\1\b» in your text editor, you can easily find them. To delete the second word, simply type in "\1" as the replacement text and click the Replace button. 

## Parentheses And Backreferences Cannot Be Used Inside Character Classes

Round brackets cannot be used inside character classes, at least not as metacharacters. When you put a round bracket in a character class, it is treated as a literal character. So the regex «[(a)b]» matches "a", "b", "(" 
and ")". Backreferences also cannot be used inside a character class. The \1 in regex like «(a)[\1b]» will be interpreted as an octal escape in most regex flavors. So this regex will match an "a" followed by either «\x01» or a «b». 

## 12. Named Capturing Groups

All modern regular expression engines support capturing groups, which are numbered from left to right, starting with one. The numbers can then be used in backreferences to match the same text again in the regular expression, or to use part of the regex match for further processing. In a complex regular expression with many capturing groups, the numbering can get a little confusing. 

## Named Capture With Python, Pcre And Php

Python's regex module was the first to offer a solution: named capture. By assigning a name to a capturing group, you can easily reference it by name. «(?P<name>group)» captures the match of «group» into the backreference "name". You can reference the contents of the group with the numbered backreference «\1» 
or the named backreference «(?P=name)». 

The open source PCRE library has followed Python's example, and offers named capture using the same syntax. The PHP preg functions offer the same functionality, since they are based on PCRE. Python's sub() function allows you to reference a named group as "\1" or "\g<name>". This does not work in PHP. In PHP, you can use double-quoted string interpolation with the $regs parameter you passed to pcre_match(): "$regs['name']". 

## Named Capture With .Net'S System.Text.Regularexpressions

The regular expression classes of the .NET framework also support named capture. Unfortunately, the Microsoft developers decided to invent their own syntax, rather than follow the one pioneered by Python. Currently, no other regex flavor supports Microsoft's version of named capture. Here is an example with two capturing groups in .NET style: «(?<first>group)(?'second'group)». As you can see, .NET offers two syntaxes to create a capturing group: one using sharp brackets, and the other using single quotes. The first syntax is preferable in strings, where single quotes may need to be escaped. The second syntax is preferable in ASP code, where the sharp brackets are used for HTML tags. You can use the pointy bracket flavor and the quoted flavors interchangeably. To reference a capturing group inside the regex, use «\k<name>» or «\k'name'». Again, you can use the two syntactic variations interchangeably. When doing a search-and-replace, you can reference the named group with the familiar dollar sign syntax: 
"${name}". Simply use a name instead of a number between the curly braces. 

## Names And Numbers For Capturing Groups

Here is where things get a bit ugly. Python and PCRE treat named capturing groups just like unnamed capturing groups, and number both kinds from left to right, starting with one. The regex 
«(a)(?P<x>b)(c)(?P<y>d)» matches "abcd" as expected. If you do a search-and-replace with this regex and the replacement "\1\2\3\4", you will get "abcd". All four groups were numbered from left to right, from one till four. Easy and logical. Things are quite a bit more complicated with the .NET framework. The regex «(a)(?<x>b)(c)(?<y>d)» 
again matches "abcd". However, if you do a search-and-replace with "$1$2$3$4" as the replacement, you will get "acbd". Probably not what you expected. 

The .NET framework *does* number named capturing groups from left to right, but numbers them *after* all the unnamed groups have been numbered. So the unnamed groups «(a)» and «(c)» get numbered first, from left to right, starting at one. Then the named groups «(?<x>b)» and «(?<y>d)» get their numbers, continuing from the unnamed groups, in this case: three. To make things simple, when using .NET's regex support, just assume that named groups do not get numbered at all, and reference them by name exclusively. To keep things compatible across regex flavors, I 
strongly recommend that you do not mix named and unnamed capturing groups at all. Either give a group a name, or make it non-capturing as in «(?:nocapture)». Non-capturing groups are more efficient, since the regex engine does not need to keep track of their matches. 

## Other Regex Flavors

EditPad Pro and PowerGREP support both the Python syntax and the .NET syntax for named capture. However, they will number named groups along with unnamed capturing groups, just like Python does. RegexBuddy also supports both Python's and Microsoft's style. RegexBuddy will convert one flavor of named capture into the other when generating source code snippets for Python, PHP/preg, PHP, or one of the .NET languages. 

None of the other regex flavors discussed in this book support named capture. 

## 13. Unicode Regular Expressions

Unicode is a character set that aims to define all characters and glyphs from all human languages, living and dead. With more and more software being required to support multiple languages, or even just any language, Unicode has been strongly gaining popularity in recent years. Using different character sets for different languages is simply too cumbersome for programmers and users. Unfortunately, Unicode brings its own requirements and pitfalls when it comes to regular expressions. Of the regex flavors discussed in this tutorial, Java, XML and the .NET framework use Unicode-based regex engines. Perl supports Unicode starting with version 5.6. PCRE can optionally be compiled with Unicode support. Note that PCRE is far less flexible in what it allows for the \p tokens, despite its name "Perlcompatible". The PHP preg functions, which are based on PCRE, support Unicode when the /u option is appended to the regular expression. RegexBuddy's regex engine is fully Unicode-based starting with version 2.0.0. RegexBuddy 1.x.x did not support Unicode at all. PowerGREP uses the same Unicode regex engine starting with version 3.0.0. Earlier versions would convert Unicode files to ANSI prior to grepping with an 8-bit (i.e. non-Unicode) regex engine. EditPad Pro supports Unicode starting with version 6.0.0. 

## Characters, Code Points And Graphemes Or How Unicode Makes A Mess Of Things

Most people would consider "‡" a single character. Unfortunately, it need not be depending on the meaning of the word "character". All Unicode regex engines discussed in this tutorial treat any single Unicode *code point* as a single character. 

When this tutorial tells you that the dot matches any single character, this translates into Unicode parlance as 
"the dot matches any single Unicode code point". In Unicode, "‡" can be encoded as two code points: U+0061 (a) followed by U+0300 (grave accent). In this situation, «.» applied to "‡" will match "a" without the accent. «^.$» will fail to match, since the string consists of two code points. «^..$» matches "‡". The Unicode code point U+0300 (grave accent) is a *combining mark*. Any code point that is not a combining mark can be followed by any number of combining marks. This sequence, like U+0061 U+0300 above, is displayed as a single *grapheme* on the screen. 

Unfortunately, "‡" can also be encoded with the single Unicode code point U+00E0 (a with grave accent). The reason for this duality is that many historical character sets encode "a with grave accent" as a single character. Unicode's designers thought it would be useful to have a one-on-one mapping with popular legacy character sets, in addition to the Unicode way of separating marks and base letters (which makes arbitrary combinations not supported by legacy character sets possible). 

## How To Match A Single Unicode Grapheme

Matching a single grapheme, whether it's encoded as a single code point, or as multiple code points using combining marks, is easy in Perl, RegexBuddy and PowerGREP: simply use «\X». You can consider «\X» the Unicode version of the dot in regex engines that use plain ASCII. There is one difference, though: «\X» always matches line break characters, whereas the dot does not match line break characters unless you enable the dot matches newline matching mode. Java and .NET unfortunately do not support «\X» (yet). Use «\P{M}\p{M}*» as a substitute. To match any number of graphemes, use «(?:\P{M}\p{M}*)+» instead of «\X+». 

## Matching A Specific Code Point

To match a specific Unicode code point, use «\uFFFF» where FFFF is the hexadecimal number of the code point you want to match. You must always specify 4 hexadecimal digits E.g. «\u00E0» matches "‡", but only when encoded as a single code point U+00E0. 

Perl and PCRE do not support the «\uFFFF» syntax. They use «\x{FFFF}» instead. You can omit leading zeros in the hexadecimal number between the curly braces. Since \x by itself is not a valid regex token, «\x{1234}» can never be confused to match \x 1234 times. It always matches the Unicode code point U+1234. «\x{1234}{5678}» will try to match code point U+1234 exactly 5678 times. In Java, the regex token «\uFFFF» only matches the specified code point, even when you turned on canonical equivalence. However, the same syntax \uFFFF is also used to insert Unicode characters into literal strings in the Java source code. Pattern.compile("\u00E0") will match both the single-code-point and doublecode-point encodings of "‡", while Pattern.compile("\\u00E0") matches only the single-code-point version. Remember that when writing a regex as a Java string literal, backslashes must be escaped. The former Java code compiles the regex «‡», while the latter compiles «\u00E0». Depending on what you're doing, the difference may be significant. JavaScript, which does not offer any Unicode support through its RegExp class, does support «\uFFFF» for matching a single Unicode code point as part of its string syntax. XML Schema does not have a regex token for matching Unicode code points. However, you can easily use XML entities like  to insert literal code points into your regular expression. 

## Unicode Character Properties

In addition to complications, Unicode also brings new possibilities. One is that each Unicode character belongs to a certain category. You can match a single character belonging to a particular category with «\p{}». You can match a single character not belonging to a particular category with «\P{}». Again, "character" really means "Unicode code point". «\p{L}» matches a single code point in the category 
"letter". If your input string is "‡" encoded as U+0061 U+0300, it matches "a" without the accent. If the input is "‡" encoded as U+00E0, it matches "‡" with the accent. The reason is that both the code points U+0061 (a) and U+00E0 (à) are in the category "letter", while U+0300 is in the category "mark". You should now understand why «\P{M}\p{M}*» is the equivalent of «\X». «\P{M}» matches a code point that is not a combining mark, while «\p{M}*» matches zero or more code points that are combining marks. 

To match a letter including any diacritics, use «\p{L}\p{M}*». This last regex will always match "‡", 
regardless of how it is encoded. The .NET Regex class and PCRE are case sensitive when it checks the part between curly braces of a \p token. «\p{Zs}» will match any kind of space character, while «\p{zs}» will throw an error. All other regex engines described in this tutorial will match the space in both cases, ignoring the case of the property between the curly braces. Still, I recommend you make a habit of using the same uppercase and lowercase combination as I did in the list of properties below. This will make your regular expressions work with all Unicode regex engines. In addition to the standard notation, «\p{L}», Java, Perl, PCRE and the JGsoft engine allow you to use the shorthand «\pL». The shorthand only works with single-letter Unicode properties. «\pLl» is not the equivalent of «\p{Ll}». It is the equivalent of «\p{L}l» which matches "Al" or "‡l" or any Unicode letter followed by a literal "l". Perl and the JGsoft engine also support the longhand «\p{Letter}». You can find a complete list of all Unicode properties below. You may omit the underscores or use hyphens or spaces instead. 

- «\p{L}» or «\p{Letter}»: any kind of letter from any language. 

o «\p{Ll}» or «\p{Lowercase_Letter}»: a lowercase letter that has an uppercase variant. 

o «\p{Lu}» or «\p{Uppercase_Letter}»: an uppercase letter that has a lowercase variant. 

o «\p{Lt}» or «\p{Titlecase_Letter}»: a letter that appears at the start of a word when only the first letter of the word is capitalized. 

o «\p{L&}» or «\p{Letter&}»: a letter that exists in lowercase and uppercase variants 
(combination of Ll, Lu and Lt). 

o «\p{Lm}» or «\p{Modifier_Letter}»: a special character that is used like a letter. 

o «\p{Lo}» or «\p{Other_Letter}»: a letter or ideograph that does not have lowercase and uppercase variants. 

- «\p{M}» or «\p{Mark}»: a character intended to be combined with another character (e.g. accents, umlauts, enclosing boxes, etc.). 

o «\p{Mn}» or «\p{Non_Spacing_Mark}»: a character intended to be combined with another character that does not take up extra space (e.g. accents, umlauts, etc.). 

o «\p{Mc}» or «\p{Spacing_Combining_Mark}»: a character intended to be combined with another character that takes up extra space (vowel signs in many Eastern languages). 

o «\p{Me}» or «\p{Enclosing_Mark}»: a character that encloses the character is is combined with (circle, square, keycap, etc.). 

- «\p{Z}» or «\p{Separator}»: any kind of whitespace or invisible separator. 

o «\p{Zs}» or «\p{Space_Separator}»: a whitespace character that is invisible, but does take up space. 

o «\p{Zl}» or «\p{Line_Separator}»: line separator character U+2028. 

o «\p{Zp}» or «\p{Paragraph_Separator}»: paragraph separator character U+2029. 

- «\p{S}» or «\p{Symbol}»: math symbols, currency signs, dingbats, box-drawing characters, etc.. 

o «\p{Sm}» or «\p{Math_Symbol}»: any mathematical symbol. o «\p{Sc}» or «\p{Currency_Symbol}»: any currency sign. 

o «\p{Sk}» or «\p{Modifier_Symbol}»: a combining character (mark) as a full character on its own. 

o «\p{So}» or «\p{Other_Symbol}»: various symbols that are not math symbols, currency signs, or combining characters. 

- «\p{N}» or «\p{Number}»: any kind of numeric character in any script. 

o «\p{Nd}» or «\p{Decimal_Digit_Number}»: a digit zero through nine in any script except ideographic scripts. 

o «\p{Nl}» or «\p{Letter_Number}»: a number that looks like a letter, such as a Roman numeral. 

o «\p{No}» or «\p{Other_Number}»: a superscript or subscript digit, or a number that is not a digit 0..9 (excluding numbers from ideographic scripts). 

- «\p{P}» or «\p{Punctuation}»: any kind of punctuation character. 

o «\p{Pd}» or «\p{Dash_Punctuation}»: any kind of hyphen or dash. 

o «\p{Ps}» or «\p{Open_Punctuation}»: any kind of opening bracket. 

o «\p{Pe}» or «\p{Close_Punctuation}»: any kind of closing bracket. 

o «\p{Pi}» or «\p{Initial_Punctuation}»: any kind of opening quote. 

o «\p{Pf}» or «\p{Final_Punctuation}»: any kind of closing quote. o «\p{Pc}» or «\p{Connector_Punctuation}»: a punctuation character such as an underscore that connects words. 

o «\p{Po}» or «\p{Other_Punctuation}»: any kind of punctuation character that is not a dash, bracket, quote or connector. 

- «\p{C}» or «\p{Other}»: invisible control characters and unused code points. 

o «\p{Cc}» or «\p{Control}»: an ASCII 0x00..0x1F or Latin-1 0x80..0x9F control character. 

o «\p{Cf}» or «\p{Format}»: invisible formatting indicator. 

o «\p{Co}» or «\p{Private_Use}»: any code point reserved for private use. 

o «\p{Cs}» or «\p{Surrogate}»: one half of a surrogate pair in UTF-16 encoding. o «\p{Cn}» or «\p{Unassigned}»: any code point to which no character has been assigned. 

## Unicode Scripts

The Unicode standard places each assigned code point (character) into one script. A script is a group of code points used by a particular human writing system. Some scripts like Thai correspond with a single human language. Other scripts like Latin span multiple languages. Some languages are composed of multiple scripts. There is no Japanese Unicode script. Instead, Unicode offers the Hiragana, Katakana, Han and Latin scripts that Japanese documents are usually composed of. 

A special script is the Common script. This script contains all sorts of characters that are common to a wide range of scripts. It includes all sorts of punctuation, whitespace and miscellaneous symbols. 

All assigned Unicode code points (those matched by «\P{Cn}») are part of exactly one Unicode script. All unassigned Unicode code points (those matched by «\p{Cn}») are not part of any Unicode script at all. Very few regular expression engines support Unicode scripts today. Of all the flavors discussed in this tutorial, only the JGsoft engine, Perl and PCRE can match Unicode scripts. Here's a complete list of all Unicode scripts: 
1. «\p{Common}» 
2. «\p{Arabic}» 
3. «\p{Armenian}» 4. «\p{Bengali}» 5. «\p{Bopomofo}» 
6. «\p{Braille}» 
7. «\p{Buhid}» 8. «\p{CanadianAboriginal}» 9. «\p{Cherokee}» 
10. «\p{Cyrillic}» 
11. «\p{Devanagari}» 12. «\p{Ethiopic}» 13. «\p{Georgian}» 
14. «\p{Greek}» 
15. «\p{Gujarati}» 
16. «\p{Gurmukhi}» 
17. «\p{Han}» 
18. «\p{Hangul}» 
19. «\p{Hanunoo}» 20. «\p{Hebrew}» 21. «\p{Hiragana}» 
22. «\p{Inherited}» 
23. «\p{Kannada}» 24. «\p{Katakana}» 25. «\p{Khmer}» 
26. «\p{Lao}» 
27. «\p{Latin}» 28. «\p{Limbu}» 29. «\p{Malayalam}» 
30. «\p{Mongolian}» 
31. «\p{Myanmar}» 32. «\p{Ogham}» 33. «\p{Oriya}» 
34. «\p{Runic}» 
35. «\p{Sinhala}» 36. «\p{Syriac}» 37. «\p{Tagalog}» 
38. «\p{Tagbanwa}» 
39. «\p{TaiLe}» 40. «\p{Tamil}» 41. «\p{Telugu}» 
42. «\p{Thaana}» 
43. «\p{Thai}» 44. «\p{Tibetan}» 45. «\p{Yi}» 
Instead of the «\p{Latin}» syntax you can also use «\p{IsLatin}». The "Is" syntax is useful for distinguishing between scripts and blocks, as explained in the next section. Unfortunately, PCRE does not support "Is" as of this writing. 

## Unicode Blocks

The Unicode standard divides the Unicode character map into different blocks or ranges of code points. 

Each block is used to define characters of a particular script like "Tibetan" or belonging to a particular group like "Braille Patterns". Most blocks include unassigned code points, reserved for future expansion of the Unicode standard. 

Note that Unicode blocks do not correspond 100% with scripts. An essential difference between blocks and scripts is that a block is a single contiguous range of code points, as listed below. Scripts consist of characters taken from all over the Unicode character map. Blocks may include unassigned code points (i.e. code points matched by «\p{Cn}»). Scripts never include unassigned code points. Generally, if you're not sure whether to use a Unicode script or Unicode block, use the script. E.g. the Currency block does not include the dollar and yen symbols. Those are found in the Basic_Latin and Latin-1_Supplement blocks instead, for historical reasons, even though both are currency symbols, and the yen symbol is not a Latin character. You should not blindly use any of the blocks listed below based on their names. Instead, look at the ranges of characters they actually match. A tool like RegexBuddy can be very helpful with this. E.g. the Unicode property «\p{Sc}» or «\p{Currency_Symbol}» would be a better choice than the Unicode block «\p{InCurrency}» when trying to find all currency symbols. 

1. «\p{InBasic_Latin}»: U+0000..U+007F 
2. «\p{InLatin-1_Supplement}»: U+0080..U+00FF 
3. «\p{InLatin_Extended-A}»: U+0100..U+017F 4. «\p{InLatin_Extended-B}»: U+0180..U+024F 5. «\p{InIPA_Extensions}»: U+0250..U+02AF 
6. «\p{InSpacing_Modifier_Letters}»: U+02B0..U+02FF 
7. «\p{InCombining_Diacritical_Marks}»: U+0300..U+036F 8. «\p{InGreek_and_Coptic}»: U+0370..U+03FF 9. «\p{InCyrillic}»: U+0400..U+04FF 
10. «\p{InCyrillic_Supplementary}»: U+0500..U+052F 
11. «\p{InArmenian}»: U+0530..U+058F 12. «\p{InHebrew}»: U+0590..U+05FF 13. «\p{InArabic}»: U+0600..U+06FF 
14. «\p{InSyriac}»: U+0700..U+074F 
15. «\p{InThaana}»: U+0780..U+07BF 16. «\p{InDevanagari}»: U+0900..U+097F 17. «\p{InBengali}»: U+0980..U+09FF 
18. «\p{InGurmukhi}»: U+0A00..U+0A7F 
19. «\p{InGujarati}»: U+0A80..U+0AFF 20. «\p{InOriya}»: U+0B00..U+0B7F 21. «\p{InTamil}»: U+0B80..U+0BFF 
22. «\p{InTelugu}»: U+0C00..U+0C7F 
23. «\p{InKannada}»: U+0C80..U+0CFF 24. «\p{InMalayalam}»: U+0D00..U+0D7F 25. «\p{InSinhala}»: U+0D80..U+0DFF 
26. «\p{InThai}»: U+0E00..U+0E7F 
27. «\p{InLao}»: U+0E80..U+0EFF 28. «\p{InTibetan}»: U+0F00..U+0FFF 29. «\p{InMyanmar}»: U+1000..U+109F 
30. «\p{InGeorgian}»: U+10A0..U+10FF 
31. «\p{InHangul_Jamo}»: U+1100..U+11FF 32. «\p{InEthiopic}»: U+1200..U+137F 33. «\p{InCherokee}»: U+13A0..U+13FF 
34. «\p{InUnified_Canadian_Aboriginal_Syllabics}»: U+1400..U+167F 
35. «\p{InOgham}»: U+1680..U+169F 36. «\p{InRunic}»: U+16A0..U+16FF 37. «\p{InTagalog}»: U+1700..U+171F 
38. «\p{InHanunoo}»: U+1720..U+173F 
39. «\p{InBuhid}»: U+1740..U+175F 40. «\p{InTagbanwa}»: U+1760..U+177F 41. «\p{InKhmer}»: U+1780..U+17FF 42. «\p{InMongolian}»: U+1800..U+18AF 43. «\p{InLimbu}»: U+1900..U+194F 
44. «\p{InTai_Le}»: U+1950..U+197F 
45. «\p{InKhmer_Symbols}»: U+19E0..U+19FF 
46. «\p{InPhonetic_Extensions}»: U+1D00..U+1D7F 
47. «\p{InLatin_Extended_Additional}»: U+1E00..U+1EFF 
48. «\p{InGreek_Extended}»: U+1F00..U+1FFF 
49. «\p{InGeneral_Punctuation}»: U+2000..U+206F 50. «\p{InSuperscripts_and_Subscripts}»: U+2070..U+209F 51. «\p{InCurrency_Symbols}»: U+20A0..U+20CF 
52. «\p{InCombining_Diacritical_Marks_for_Symbols}»: U+20D0..U+20FF 
53. «\p{InLetterlike_Symbols}»: U+2100..U+214F 54. «\p{InNumber_Forms}»: U+2150..U+218F 55. «\p{InArrows}»: U+2190..U+21FF 
56. «\p{InMathematical_Operators}»: U+2200..U+22FF 
57. «\p{InMiscellaneous_Technical}»: U+2300..U+23FF 58. «\p{InControl_Pictures}»: U+2400..U+243F 59. «\p{InOptical_Character_Recognition}»: U+2440..U+245F 
60. «\p{InEnclosed_Alphanumerics}»: U+2460..U+24FF 
61. «\p{InBox_Drawing}»: U+2500..U+257F 62. «\p{InBlock_Elements}»: U+2580..U+259F 63. «\p{InGeometric_Shapes}»: U+25A0..U+25FF 
64. «\p{InMiscellaneous_Symbols}»: U+2600..U+26FF 
65. «\p{InDingbats}»: U+2700..U+27BF 66. «\p{InMiscellaneous_Mathematical_Symbols-A}»: U+27C0..U+27EF 67. «\p{InSupplemental_Arrows-A}»: U+27F0..U+27FF 
68. «\p{InBraille_Patterns}»: U+2800..U+28FF 
69. «\p{InSupplemental_Arrows-B}»: U+2900..U+297F 70. «\p{InMiscellaneous_Mathematical_Symbols-B}»: U+2980..U+29FF 71. «\p{InSupplemental_Mathematical_Operators}»: U+2A00..U+2AFF 
72. «\p{InMiscellaneous_Symbols_and_Arrows}»: U+2B00..U+2BFF 
73. «\p{InCJK_Radicals_Supplement}»: U+2E80..U+2EFF 74. «\p{InKangxi_Radicals}»: U+2F00..U+2FDF 75. «\p{InIdeographic_Description_Characters}»: U+2FF0..U+2FFF 
76. «\p{InCJK_Symbols_and_Punctuation}»: U+3000..U+303F 
77. «\p{InHiragana}»: U+3040..U+309F 
78. «\p{InKatakana}»: U+30A0..U+30FF 79. «\p{InBopomofo}»: U+3100..U+312F 
80. «\p{InHangul_Compatibility_Jamo}»: U+3130..U+318F 
81. «\p{InKanbun}»: U+3190..U+319F 82. «\p{InBopomofo_Extended}»: U+31A0..U+31BF 83. «\p{InKatakana_Phonetic_Extensions}»: U+31F0..U+31FF 
84. «\p{InEnclosed_CJK_Letters_and_Months}»: U+3200..U+32FF 
85. «\p{InCJK_Compatibility}»: U+3300..U+33FF 86. «\p{InCJK_Unified_Ideographs_Extension_A}»: U+3400..U+4DBF 87. «\p{InYijing_Hexagram_Symbols}»: U+4DC0..U+4DFF 
88. «\p{InCJK_Unified_Ideographs}»: U+4E00..U+9FFF 
89. «\p{InYi_Syllables}»: U+A000..U+A48F 90. «\p{InYi_Radicals}»: U+A490..U+A4CF 91. «\p{InHangul_Syllables}»: U+AC00..U+D7AF 
92. «\p{InHigh_Surrogates}»: U+D800..U+DB7F 
93. «\p{InHigh_Private_Use_Surrogates}»: U+DB80..U+DBFF 94. «\p{InLow_Surrogates}»: U+DC00..U+DFFF 95. «\p{InPrivate_Use_Area}»: U+E000..U+F8FF 
96. «\p{InCJK_Compatibility_Ideographs}»: U+F900..U+FAFF 
97. «\p{InAlphabetic_Presentation_Forms}»: U+FB00..U+FB4F 
98. «\p{InArabic_Presentation_Forms-A}»: U+FB50..U+FDFF 
99. «\p{InVariation_Selectors}»: U+FE00..U+FE0F 
100. «\p{InCombining_Half_Marks}»: U+FE20..U+FE2F 
101. «\p{InCJK_Compatibility_Forms}»: U+FE30..U+FE4F 102. «\p{InSmall_Form_Variants}»: U+FE50..U+FE6F 103. «\p{InArabic_Presentation_Forms-B}»: U+FE70..U+FEFF 
104. «\p{InHalfwidth_and_Fullwidth_Forms}»: U+FF00..U+FFEF 
105. «\p{InSpecials}»: U+FFF0..U+FFFF 
Not all Unicode regex engines use the same syntax to match Unicode blocks. Perl and use the 
«\p{InBlock}» syntax as listed above. .NET and XML use «\p{IsBlock}» instead. The JGsoft engine supports both notations. I recommend you use the "In" notation if your regex engine supports it. "In" can only be used for Unicode blocks, while "Is" can also be used for Unicode properties and scripts, depending on the regular expression flavor you're using. By using "In", it's obvious you're matching a block and not a similarly named property or script. In .NET and XML, you must omit the underscores but keep the hyphens in the block names. E.g. Use «\p{IsLatinExtended-A}» instead of «\p{InLatin_Extended-A}». Perl and Java allow you to use an underscore, hyphen, space or nothing for each underscore or hyphen in the block's name. .NET and XML 
also compare the names case sensitively, while Perl and Java do not. «\p{islatinextended-a}» throws an error in .NET, while «\p{inlatinextended-a}» works fine in Perl and Java. The JGsoft engine supports all of the above notations. You can use "In" or "Is", ignore differences in upper and lower case, and use spaces, underscores and hyphens as you like. This way you can keep using the syntax of your favorite programming language, and have it work as you'd expect in PowerGREP or EditPad Pro. The actual names of the blocks are the same in all regular expression engines. The block names are defined in the Unicode standard. PCRE does not support Unicode blocks. 

## Alternative Unicode Regex Syntax

Unicode is a relatively new addition to the world of regular expressions. As you guessed from my explanations of different notations, different regex engine designers unfortunately have different ideas about the syntax to use. Perl and Java even support a few additional alternative notations that you may encounter in regular expressions created by others. I recommend against using these notations in your own regular expressions, to maintain clarity and compatibility with other regex flavors, and understandability by people more familiar with other flavors. 

If you are just getting started with Unicode regular expressions, you may want to skip this section until later, to avoid confusion (if the above didn't confuse you already). In Perl and PCRE regular expressions, you may encounter a Unicode property like «\p{^Lu}» or 
«\p{^Letter}». These are negated properties identical to «\P{Lu}» or «\P{Letter}». Since very few regex flavors support the «\p{^L}» notation, and all Unicode-compatible regex flavors (including Perl and PCRE) support «\P{L}», I strongly recommend you use the latter syntax. Perl (but not PCRE) and Java support the «\p{IsL}» notation, prefixing one-letter and two-letter Unicode property notations with "Is". Since very few regex flavors support the «\p{IsL}» notation, and all Unicodecompatible regex flavors (including Perl and Java) support «\p{L}», I strongly recommend you use the latter syntax. Perl and Java allow you to omit the "In" when matching Unicode blocks, so you can write «\p{Arrows}» instead of «\p{InArrows}». Perl can also match Unicode scripts, and some scripts like "Hebrew" have the same name as a Unicode block. In that situation, Perl will match the Hebrew script instead of the Hebrew block when you write «\p{Hebrew}». While there are no Unicode properties with the same names as blocks, the property «\p{Currency_Symbol}» is confusingly similar to the block «\p{Currency}». As I explained in the section on Unicode blocks, the characters they match are quite different. To avoid all such confusion, I strongly recommend you use the "In" syntax for blocks, the "Is" syntax for scripts (if supported), and the shorthand syntax «\p{Lu}» for properties. 

Again, the JGsoft engine supports all of the above oddball notations. This is only done to allow you to copy and paste regular expressions and have them work as they do in Perl or Java. You should consider these notations deprecated. 

## Do You Need To Worry About Different Encodings?

While you should always keep in mind the pitfalls created by the different ways in which accented characters can be encoded, you don't always have to worry about them. If you know that your input string and your regex use the same style, then you don't have to worry about it at all. This process is called Unicode normalization. All programming languages with native Unicode support, such as Java, C\# and VB.NET, have library routines for normalizing strings. If you normalize both the subject and regex before attempting the match, there won't be any inconsistencies. If you are using Java, you can pass the CANON_EQ flag as the second parameter to Pattern.compile(). This tells the Java regex engine to consider *canonically equivalent* characters as identical. E.g. the regex «‡» encoded as U+00E0 will match "‡" encoded as U+0061 U+0300, and vice versa. None of the other regex engines currently support canonical equivalence while matching. 

If you type the à key on the keyboard, all word processors that I know of will insert the code point U+00E0 into the file. So if you're working with text that you typed in yourself, any regex that you type in yourself will match in the same way. Finally, if you're using PowerGREP to search through text files encoded using a traditional Windows (often called "ANSI") or ISO-8859 code page, PowerGREP will always use the one-on-one substitution. Since all the Windows or ISO-8859 code pages encode accented characters as a single code point, all software that I know of will use a single Unicode code point for each character when converting the file to Unicode. 

## 14. Regex Matching Modes

Most regular expression engines discussed in this tutorial support the following four matching modes: 
- /i makes the regex match case insensitive. - /s enables "single-line mode". In this mode, the dot matches newlines. - /m enables "multi-line mode". In this mode, the caret and dollar match before and after newlines in the subject string. 

- /x enables "free-spacing mode". In this mode, whitespace between regex tokens is ignored, and an unescaped \# starts a comment. 

Two languages that don't support all of the above three are JavaScript and Ruby. Some regex flavors also have additional modes or options that have single letter equivalents. These are very implementationdependent. 

Most tools that support regular expressions have checkboxes or similar controls that you can use to turn these modes on or off. Most programming languages allow you to pass option flags when constructing the regex object. E.g. in Perl, m/regex/i turns on case insensitivity, while Pattern.compile("regex", Pattern.CASE_INSENSITIVE) does the same in Java. 

## Specifying Modes Inside The Regular Expression

Sometimes, the tool or language does not provide the ability to specify matching options. E.g. the handy String.matches() method in Java does not take a parameter for matching options like Pattern.compile() does. In that situation, you can add a mode modifier to the start of the regex. E.g. (?i) turns on case insensitivity, while (?ism) turns on all three options. 

## Turning Modes On And Off For Only Part Of The Regular Expression

Modern regex flavors allow you to apply modifiers to only part of the regular expression. If you insert the modifier (?ism) in the middle of the regex, the modifier only applies to the part of the regex to the right of the modifier. You can turn off modes by preceding them with a minus sign. All modes after the minus sign will be turned off. E.g. (?i-sm) turns on case insensitivity, and turns off both single-line mode and multiline mode. 

Not all regex flavors support this. JavaScript and Python apply all mode modifiers to the entire regular expression. They don't support the (?-ismx) syntax, since turning off an option is pointless when mode modifiers apply to the whole regular expressions. All options are off by default. 

You can quickly test how the regex flavor you're using handles mode modifiers. The regex «(?i)te(?-
i)st» should match "test" and "TEst", but not "teST" or "TEST". 

## Modifier Spans

Instead of using two modifiers, one to turn an option on, and one to turn it off, you use a modifier span. 

«(?i)ignorecase(?-i)casesensitive(?i)ignorecase» is equivalent to «(?i)ignorecase(?-
i:casesensitive)ignorecase». You have probably noticed the resemblance between the modifier span and the non-capturing group «(?:group)». Technically, the non-capturing group is a modifier span that does not change any modifiers. It is obvious that the modifier span does not create a backreference. Modifier spans are supported by all regex flavors that allow you to use mode modifiers in the middle of the regular expression, and by those flavors only. These include the JGsoft engine, .NET, Java, Perl and PCRE. 

## 15. Possessive Quantifiers

When discussing the repetition operators or quantifiers, I explained the difference between greedy and lazy repetition. Greediness and laziness determine the order in which the regex engine tries the possible permutations of the regex pattern. A greedy quantifier will first try to repeat the token as many times as possible, and gradually give up matches as the engine backtracks to find an overall match. A lazy quantifier will first repeat the token as few times as required, and gradually expand the match as the engine backtracks through the regex to find an overall match. 

Because greediness and laziness change the order in which permutations are tried, they can change the overall regex match. However, they do not change the fact that the regex engine will backtrack to try all possible permutations of the regular expression in case no match can be found. Possessive quantifiers are a way to prevent the regex engine from trying all permutations. This is primarily useful for performance reasons. You can also use possessive quantifiers to eliminate certain matches. 

## How Possessive Quantifiers Work

Several modern regular expression flavors, including the JGsoft, Java and PCRE have a third kind of quantifier: the possessive quantifier. Like a greedy quantifier, a possessive quantifier will repeat the token as many times as possible. Unlike a greedy quantifier, it will not give up matches as the engine backtracks. With a possessive quantifier, the deal is all or nothing. You can make a quantifier possessive by placing an extra +
after it. E.g. «*» is greedy, «*?» is lazy, and «*+» is possessive. «++», «?+» and «{n,m}+» are all possessive as well. Let's see what happens if we try to match «"[^"]*+"» against ""abc"". The «"» matches the """. «[^"]» matches "a", "b" and "c" as it is repeated by the star. The final «"» then matches the final """ and we found an overall match. In this case, the end result is the same, whether we use a greedy or possessive quantifier. 

There is a slight performance increase though, because the possessive quantifier doesn't have to remember any backtracking positions. The performance increase can be significant in situations where the regex fails. If the subject is ""abc" (no closing quote), the above matching process will happen in the same way, except that the second «"» fails. 

When using a possessive quantifier, there are no steps to backtrack to. The regular expression does not have any alternation or non-possessive quantifiers that can give up part of their match to try a different permutation of the regular expression. So the match attempt fails immediately when the second «"» fails. Had we used a greedy quantifier instead, the engine would have backtracked. After the «"» failed at the end of the string, the «[^"]*» would give up one match, leaving it with "ab". The «"» would then fail to match "c". «[^"]*» backtracks to just "a", and «"» fails to match "b". Finally, «[^"]*» backtracks to match zero characters, and «"» fails "a". Only at this point have all backtracking positions been exhausted, and does the engine give up the match attempt. Essentially, this regex performs as many needless steps as there are characters following the unmatched opening quote. 

## When Possessive Quantifiers Matter

The main practical benefit of possessive quantifiers is to speed up your regular expression. In particular, possessive quantifiers allow your regex to fail faster. In the above example, when the closing quote fails to match, we *know* the regular expression couldn't have possibly skipped over a quote. So there's no need to backtrack and check for the quote. We make the regex engine aware of this by making the quantifier possessive. In fact, some engines, including the JGsoft engine detect that «[^"]*» and «"» are mutually exclusive when compiling your regular expression, and automatically make the star possessive. 

Now, linear backtracking like a regex with a single quantifier does is pretty fast. It's unlikely you'll notice the speed difference. However, when you're nesting quantifiers, a possessive quantifier may save your day. 

Nesting quantifiers means that you have one or more repeated tokens inside a group, and the group is also repeated. That's when catastrophic backtracking often rears its ugly head. In such cases, you'll depend on possessive quantifiers and/or atomic grouping to save the day. 

## Possessive Quantifiers Can Change The Match Result

Using possessive quantifiers can change the result of a match attempt. Since no backtracking is done, and matches that would require a greedy quantifier to backtrack will not be found with a possessive quantifier. 

E.g. «".*"» will match ""abc"" in ""abc"x", but «".*+"» will not match this string at all. In both regular expressions, the first «"» will match the first """ in the string. The repeated dot then matches the remainder of the string "abc"x". The second «"» then fails to match at the end of the string. Now, the paths of the two regular expressions diverge. The possessive dot-star wants it all. No backtracking is done. Since the «"» failed, there are no permutations left to try, and the overall match attempt fails. The greedy dot-star, while initially grabbing everything, is willing to give back. It will backtrack one character at a time. Backtracking to "abc"", «"» fails to match "x". Backtracking to "abc", «"» matches """. An overall match ""abc"" was found. 

Essentially, the lesson here is that when using possessive quantifiers, you need to make sure that whatever you're applying the possessive quantifier to should not be able to match what should follow it. The problem in the above example is that the dot also matches the closing quote. This prevents us from using a possessive quantifier. The negated character class in the previous section cannot match the closing quote, so we can make it possessive. 

## Using Atomic Grouping Instead Of Possessive Quantifiers

Technically, possessive quantifiers are a notational convenience to place an atomic group around a single quantifier. All regex flavors that support possessive quantifiers also support atomic grouping. But not all regex flavors that support atomic grouping support possessive quantifiers. With those flavors, you can achieve the exact same results using an atomic group. Basically, instead of «X*+», write «(>X*)». It is important to notice that both the quantified token X and the quantifier are inside the atomic group. Even if X is a group, you still need to put an extra atomic group around it to achieve the same effect. «(?:a|b)*+» is equivalent to «(?>(?:a|b)*)» but not to «(?>a|b)*». 

The latter is a valid regular expression, but it won't have the same effect when used as part of a larger regular expression. E.g. «(?:a|b)*+b» and «(?>(?:a|b)*)b» both fail to match "b". «a|b» will match the "b". The star is satisfied, and the fact that it's possessive or the atomic group will cause the star to forget all its backtracking positions. The second «b» in the regex has nothing left to match, and the overall match attempt fails. 

In the regex «(?>a|b)*b», the atomic group forces the alternation to give up its backtracking positions. I.e. if an "a" is matched, it won't come back to try «b» if the rest of the regex fails. Since the star is outside of the group, it is a normal, greedy star. When the second «b» fails, the greedy star will backtrack to zero iterations. Then, the second «b» matches the "b" in the subject string. This distinction is particularly important when converting a regular expression written by somebody else using possessive quantifiers to a regex flavor that doesn't have possessive quantifiers. You could, of course, let a tool like RegexBuddy do the job for you. 

## 16. Atomic Grouping

An atomic group is a group that, when the regex engine exits from it, automatically throws away all backtracking positions remembered by any tokens inside the group. Atomic groups are non-capturing. The syntax is «(?>group)». Lookaround groups are also atomic. Atomic grouping is supported by most modern regular expression flavors, including the JGsoft flavor, Java, PCRE, .NET, Perl and Ruby. The first three of these also support possessive quantifiers, which are essentially a notational convenience for atomic grouping. An example will make the behavior of atomic groups. The regular expression «a(bc|b)c» (capturing group) matches "abcc" and "abc". The regex «a(?>bc|b)c» (atomic group) matches "abcc" but not "abc". When applied to "abc", both regexes will match «a» to "a", «bc» to "bc", and then «c» will fail to match at the end of the string. Here there paths diverge. The regex with the capturing group has remembered a backtracking position for the alternation. The group will give up its match, «b» then matches "b" and «c» matches "c". Match found! The regex with the atomic group, however, exited from an atomic group after «bc» was matched. At that point, all backtracking positions for tokens inside the group are discarded. In this example, the alternation's option to try «b» at the second position in the string is discarded. As a result, when «c» fails, the regex engine has no alternatives left to try. 

Of course, the above example isn't very useful. But it does illustrate very clearly how atomic grouping eliminates certain matches. Or more importantly, it eliminates certain match attempts. 

## Regex Optimization Using Atomic Grouping

Consider the regex «\b(integer|insert|in)\b» and the subject "integers". Obviously, because of the word boundaries, these don't match. What's not so obvious is that the regex engine will spend quite some effort figuring this out. «\b» matches at the start of the string, and «integer» matches "integer". The regex engine makes note that there are to more alternatives in the group, and continues with «\b». This fails to match between the "r" and "s". So the engine backtracks to try the second alternative inside the group. The second alternative matches "in", but then fails to match «s». So the engine backtracks once more to the third alternative. «in» matches "in". «\b» fails between the "n" and "t" this time. The regex engine has no more remembered backtracking positions, so it declares failure. This is quite a lot of work to figure out "integers" isn't in our list of words. We can optimize this by telling the regular expression engine that if it can't match «\b» after it matched "integer", then it shouldn't bother trying any of the other words. The word we've encountered in the subject string is a longer word, and it isn't in our list. We can do this my turning the capturing group into an atomic group: «\b(?>integer|insert|in)\b». Now, when «integer» matches, the engine exits from an atomic group, and throws away the backtracking positions it stored for the alternation. When «\b» fails, the engine gives up immediately. This savings can be significant when scanning a large file for a long list of keywords. This savings will be vital when your alternatives contain repeated tokens (not to mention repeated groups) that lead to catastrophic backtracking. Don't be too quick to make all your groups atomic. As we saw in the first example above, atomic grouping can exclude valid matches too. Compare how «\b(?>integer|insert|in)\b» and 
«\b(?>in|integer|insert)\b» behave when applied to "insert". The former regex matches, while the latter fails. If the groups weren't atomic, both regexes would match. Remember that alternation tries its alternatives from left to right. If the second regex matches "in", it won't try the two other alternatives due to the atomic group. 

## 17. Lookahead And Lookbehind Zero-Width Assertions

Perl 5 introduced two very powerful constructs: "lookahead" and "lookbehind". Collectively, these are called 
"lookaround". They are also called "zero-width assertions". They are zero-width just like the start and end of line, and start and end of word anchors that I already explained. The difference is that lookarounds will actually match characters, but then give up the match and only return the result: match or no match. That is why they are called "assertions". They do not consume characters in the string, but only assert whether a match is possible or not. Lookarounds allow you to create regular expressions that are impossible to create without them, or that would get very longwinded without them. 

## Positive And Negative Lookahead

Negative lookahead is indispensable if you want to match something not followed by something else. When explaining character classes, I already explained why you cannot use a negated character class to match a "q" not followed by a "u". Negative lookahead provides the solution: «q(?!u)». The negative lookahead construct is the pair of round brackets, with the opening bracket followed by a question mark and an exclamation point. Inside the lookahead, we have the trivial regex «u». 

Positive lookahead works just the same. «q(?=u)» matches a q that is followed by a u, without making the u part of the match. The positive lookahead construct is a pair of round brackets, with the opening bracket followed by a question mark and an equals sign. You can use any regular expression inside the lookahead. (Note that this is not the case with lookbehind. I will explain why below.) Any valid regular expression can be used inside the lookahead. If it contains capturing parentheses, the backreferences will be saved. Note that the lookahead itself does not create a backreference. So it is not included in the count towards numbering the backreferences. If you want to store the match of the regex inside a backreference, you have to put capturing parentheses around the regex inside the lookahead, like this: «(?=(regex))». The other way around will not work, because the lookahead will already have discarded the regex match by the time the backreference is to be saved. 

## Regex Engine Internals

First, let's see how the engine applies «q(?!u)» to the string "Iraq". The first token in the regex is the literal «q». As we already know, this will cause the engine to traverse the string until the "q" in the string is matched. 

The position in the string is now the void behind the string. The next token is the lookahead. The engine takes note that it is inside a lookahead construct now, and begins matching the regex inside the lookahead. So the next token is «u». This does not match the void behind the string. The engine notes that the regex inside the lookahead failed. Because the lookahead is negative, this means that the lookahead has successfully matched at the current position. At this point, the entire regex has matched, and "q" is returned as the match. Let's try applying the same regex to "quit". «q» matches "q". The next token is the «u» inside the lookahead. 

The next character is the "u". These match. The engine advances to the next character: "i". However, it is done with the regex inside the lookahead. The engine notes success, and discards the regex match. This causes the engine to step back in the string to "u". Because the lookahead is negative, the successful match inside it causes the lookahead to fail. Since there are no other permutations of this regex, the engine has to start again at the beginning. Since «q» cannot match anywhere else, the engine reports failure. Let's take one more look inside, to make sure you understand the implications of the lookahead. Let's apply 
«q(?=u)i» to "quit". I have made the lookahead positive, and put a token after it. Again, «q» matches "q" and «u» matches "u". Again, the match from the lookahead must be discarded, so the engine steps back from "i" in the string to "u". The lookahead was successful, so the engine continues with «i». But «i» cannot match "u". So this match attempt fails. All remaining attempts will fail as well, because there are no more q's in the string. 

## Positive And Negative Lookbehind

Lookbehind has the same effect, but works backwards. It tells the regex engine to temporarily step backwards in the string, to check if the text inside the lookbehind can be matched there. «(?<!a)b» matches a "b" that is not preceded by an "a", using negative lookbehind. It will not match "cab", but will match the "b" (and only the "b") in "bed" or "debt". «(?<=a)b» (positive lookbehind) matches the "b" (and only the "b") in 
"cab", but does not match "bed" or "debt". 

The construct for positive lookbehind is «(?<=text)»: a pair of round brackets, with the opening bracket followed by a question mark, "less than" symbol and an equals sign. Negative lookbehind is written as 
«(?<!text)», using an exclamation point instead of an equals sign. 

## More Regex Engine Internals

Let's apply «(?<=a)b» to "thingamabob". The engine starts with the lookbehind and the first character in the string. In this case, the lookbehind tells the engine to step back one character, and see if an "a" can be matched there. The engine cannot step back one character because there are no characters before the "t". So the lookbehind fails, and the engine starts again at the next character, the "h". (Note that a negative lookbehind would have succeeded here.) Again, the engine temporarily steps back one character to check if an "a" can be found there. It finds a "t", so the positive lookbehind fails again. The lookbehind continues to fail until the regex reaches the "m" in the string. The engine again steps back one character, and notices that the "a" can be matched there. The positive lookbehind matches. Because it is zero-width, the current position in the string remains at the "m". The next token is «b», which cannot match here. The next character is the second "a" in the string. The engine steps back, and finds out that the "m" does not match «a». The next character is the first "b" in the string. The engine steps back and finds out that "a" satisfies the lookbehind. «b» matches "b", and the entire regex has been matched successfully. It matches one character: the first "b" in the string. 

## Important Notes About Lookbehind

The good news is that you can use lookbehind anywhere in the regex, not only at the start. If you want to find a word not ending with an "s", you could use «\b\w+(?<!s)\b». This is definitely not the same as «\b\w+[^s]\b». When applied to "John's", the former will match "John" and the latter "John'" (including the apostrophe). I will leave it up to you to figure out why. (Hint: «\b» matches between the apostrophe and the "s"). The latter will also not match single-letter words like "a" or "I". The correct regex without using lookbehind is «\b\w*[^s\W]\b» (star instead of plus, and \W in the character class). 

Personally, I find the lookbehind easier to understand. The last regex, which works correctly, has a double negation (the \W in the negated character class). Double negations tend to be confusing to humans. Not to regex engines, though. The bad news is that most regex flavors do not allow you to use just any regex inside a lookbehind, because they cannot apply a regular expression backwards. Therefore, the regular expression engine needs to be able to figure out how many steps to step back before checking the lookbehind. Therefore, many regex flavors, including those used by Perl and Python, only allow fixed-length strings. You can use any regex of which the length of the match can be predetermined. This means you can use literal text and character classes. You cannot use repetition or optional items. You can use alternation, but only if all options in the alternation have the same length. Some regex flavors, like PCRE and Java support the above, plus alternation with strings of different lengths. Each part of the alternation must still have a finite maximum length. This means you can still not use the star or plus, but you can use the question mark and the curly braces with the max parameter specified. These regex flavors recognize the fact that finite repetition can be rewritten as an alternation of strings with different, but fixed lengths. Unfortunately, the JDK 1.4 and 1.5 have some bugs when you use alternation inside lookbehind. These were fixed in JDK 1.6. The only regex engines that allow you to use a full regular expression inside lookbehind are the JGsoft engine and the .NET framework RegEx classes. Finally, flavors like JavaScript, Ruby and Tcl do not support lookbehind at all, even though they do support lookahead. 

## Lookaround Is Atomic

The fact that lookaround is zero-width automatically makes it atomic. As soon as the lookaround condition is satisfied, the regex engine forgets about everything inside the lookaround. It will not backtrack inside the lookaround to try different permutations. The only situation in which this makes any difference is when you use capturing groups inside the lookaround. Since the regex engine does not backtrack into the lookaround, it will not try different permutations of the capturing groups. For this reason, the regex «(?=(\d+))\w+\1» will never match "123x12". First the lookaround captures "123" into «\1». «\w+» then matches the whole string and backtracks until it matches only "1". Finally, «\w+» fails since «\1» cannot be matched at any position. Now, the regex engine has nothing to backtrack to, and the overall regex fails. The backtracking steps created by «\d+» have been discarded. It never gets to the point where the lookahead captures only "12". Obviously, the regex engine does try further positions in the string. If we change the subject string, the regex «(?=(\d+))\w+\1» will match "56x56" in "456x56". If you don't use capturing groups inside lookaround, then all this doesn't matter. Either the lookaround condition can be satisfied or it cannot be. In how many ways it can be satisfied is irrelevant. 

## 18. Testing The Same Part Of A String For More Than One Requirement

Lookaround, which I introduced in detail in the previous topic, is a very powerful concept. Unfortunately, it is often underused by people new to regular expressions, because lookaround is a bit confusing. The confusing part is that the lookaround is zero-width. So if you have a regex in which a lookahead is followed by another piece of regex, or a lookbehind is preceded by another piece of regex, then the regex will traverse part of the string twice. 

To make this clear, I would like to give you another, a bit more practical example. Let's say we want to find a word that is six letters long and contains the three subsequent letters "cat". Actually, we can match this without lookaround. We just specify all the options and hump them together using alternation: «cat\w{3}|\wcat\w{2}|\w{2}cat\w|\w{3}cat». Easy enough. But this method gets unwieldy if you want to find any word between 6 and 12 letters long containing either "cat", "dog" or "mouse". 

## Lookaround To The Rescue

In this example, we basically have two requirements for a successful match. First, we want a word that is 6 letters long. Second, the word we found must contain the word "cat". Matching a 6-letter word is easy with «\b\w{6}\b». Matching a word containing "cat" is equally easy: 
«\b\w*cat\w*\b». Combining the two, we get: «(?=\b\w{6}\b)\b\w*cat\w*\b» . Easy! Here's how this works. At each character position in the string where the regex is attempted, the engine will first attempt the regex inside the positive lookahead. This sub-regex, and therefore the lookahead, matches only when the current character position in the string is at the start of a 6-letter word in the string. If not, the lookahead will fail, and the engine will continue trying the regex from the start at the next character position in the string. 

The lookahead is zero-width. So when the regex inside the lookahead has found the 6-letter word, the current position in the string is still at the beginning of the 6-letter word. At this position will the regex engine attempt the remainder of the regex. Because we already know that a 6-letter word can be matched at the current position, we know that «\b» matches and that the first «\w*» will match 6 times. The engine will then backtrack, reducing the number of characters matched by «\w*», until «cat» can be matched. If «cat» cannot be matched, the engine has no other choice but to restart at the beginning of the regex, at the next character position in the string. This is at the second letter in the 6-letter word we just found, where the lookahead will fail, causing the engine to advance character by character until the next 6-letter word. If «cat» can be successfully matched, the second «\w*» will consume the remaining letters, if any, in the 6letter word. After that, the last «\b» in the regex is guaranteed to match where the second «\b» inside the lookahead matched. Our double-requirement-regex has matched successfully. 

## Optimizing Our Solution

While the above regex works just fine, it is not the most optimal solution. This is not a problem if you are just doing a search in a text editor. But optimizing things is a good idea if this regex will be used repeatedly and/or on large chunks of data in an application you are developing. 

You can discover these optimizations by yourself if you carefully examine the regex and follow how the regex engine applies it, as I did above. I said the third and last «\b» are guaranteed to match. Since it is zero-width, and therefore does not change the result returned by the regex engine, we can remove them, leaving: «(?=\b\w{6}\b)\w*cat\w*». Though the last «\w*» is also guaranteed to match, we cannot remove it because it adds characters to the regex match. Remember that the lookahead discards its match, so it does not contribute to the match returned by the regex engine. If we omitted the «\w*», the resulting match would be the start of a 6-letter word containing "cat", up to and including "cat", instead of the entire word. But we can optimize the first «\w*». As it stands, it will match 6 letters and then backtrack. But we know that in a successful match, there can never be more than 3 letters before "cat". So we can optimize this to 
«\w{0,3}». Note that making the asterisk lazy would not have optimized this sufficiently. The lazy asterisk would find a successful match sooner, but if a 6-letter word does not contain "cat", it would still cause the regex engine to try matching "cat" at the last two letters, at the last single letter, and even at one character beyond the 6-letter word. So we have «(?=\b\w{6}\b)\w{0,3}cat\w*». One last, minor, optimization involves the first «\b». Since it is zero-width itself, there's no need to put it inside the lookahead. So the final regex is: 
«\b(?=\w{6}\b)\w{0,3}cat\w*» . 

## A More Complex Problem

So, what would you use to find any word between 6 and 12 letters long containing either "cat", "dog" or 
"mouse"? Again we have two requirements, which we can easily combine using a lookahead: « \b(?=\w{6,12}\b)\w{0,9}(cat|dog|mouse)\w*» . Very easy, once you get the hang of it. This regex will also put "cat", "dog" or "mouse" into the first backreference. 

## 19. Continuing At The End Of The Previous Match

The anchor «\G» matches at the position where the previous match ended. During the first match attempt, 
«\G» matches at the start of the string in the way «\A» does. 

Applying «\G\w» to the string "test string" matches "t". Applying it again matches "e". The 3rd attempt yields "s" and the 4th attempt matches the second "t" in the string. The fifth attempt fails. During the fifth attempt, the only place in the string where «\G» matches is after the second t. But that position is not followed by a word character, so the match fails. 

## End Of The Previous Match Vs. Start Of The Match Attempt

With some regex flavors or tools, «\G» matches at the start of the match attempt, rather than at the end of the previous match result. This is the case with EditPad Pro, where «\G» matches at the position of the text cursor, rather than the end of the previous match. When a match is found, EditPad Pro will select the match, and move the text cursor to the end of the match. The result is that «\G» matches at the end of the previous match result only when you do not move the text cursor between two searches. All in all, this makes a lot of sense in the context of a text editor. 

## \G Magic With Perl

In Perl, the position where the last match ended is a "magical" value that is remembered separately for each string variable. The position is not associated with any regular expression. This means that you can use «\G» to make a regex continue in a subject string where another regex left off. If a match attempt fails, the stored position for «\G» is reset to the start of the string. To avoid this, specify the continuation modifier /c. 

All this is very useful to make several regular expressions work together. E.g. you could parse an HTML file in the following fashion: 
while ($string =~ m/</g) { if ($string =~ m/\GB>/c) { \# Bold } elsif ($string =~ m/\GI>/c) { \# Italics } else { \# ...etc... } }
The regex in the while loop searches for the tag's opening bracket, and the regexes inside the loop check which tag we found. This way you can parse the tags in the file in the order they appear in the file, without having to write a single big regex that matches all tags you are interested in. 

# \G In Other Programming Languages

This flexibility is not available with most other programming languages. E.g. in Java, the position for «\G» is remembered by the Matcher object. The Matcher is strictly associated with a single regular expression and a single subject string. What you can do though is to add a line of code to make the match attempt of the second Matcher start where the match of the first Matcher ended. «\G» will then match at this position. The «\G» token is supported by the JGsoft engine, .NET, Java, Perl and PCRE. 

# 20. If-Then-Else Conditionals In Regular Expressions

A special construct «(?ifthen|else)» allows you to create conditional regular expressions. If the if part evaluates to true, then the regex engine will attempt to match the *then* part. Otherwise, the *else* part is attempted instead. The syntax consists of a pair of round brackets. The opening bracket must be followed by a question mark, immediately followed by the if part, immediately followed by the *then* part. This part can be followed by a vertical bar and the *else* part. You may omit the *else* part, and the vertical bar with it. For the if part, you can use the lookahead and lookbehind constructs. Using positive lookahead, the syntax becomes «(?(?=regex)then|else)». Because the lookahead has its own parentheses, the if and *then* parts are clearly separated. Remember that the lookaround constructs do not consume any characters. If you use a lookahead as the if part, then the regex engine will attempt to match the then or *else part* (depending on the outcome of the lookahead) at the same position where the if was attempted. Alternatively, you can check in the if part whether a capturing group has taken part in the match thus far. 

Place the number of the capturing group inside round brackets, and use that as the if part. Note that although the syntax for a conditional check on a backreference is the same as a number inside a capturing groups, no capturing groups is created. The number and the brackets are part of the if-then-else syntax started with «(?». 

For the *then* and *else*, you can use any regular expression. If you want to use alternation, you will have to group the then or *else* together using parentheses, like in «(?(?=condition)(then1|then2|then3)|(else1|else2|else3))». Otherwise, there is no need to use parentheses around the *then* and *else* parts. 

The regex «(a)?b(?(1)c|d)» matches "bd" and "abc". It does not match "bc", but does match "bd" in "abd". Let's see how this regular expression works on each of these four subject strings. When applied to "bd", «a» fails to match. Since the capturing group containing «a» is optional, the engine continues with «b» at the start of the subject string. Since the whole group was optional, the group did not take part in the match. Any subsequent backreference to it like «\1» will fail. Note that «(a)?» is very different from «(a?)». In the former regex, the capturing group does not take part in the match if «a» fails, and backreferences to the group will fail. In the latter group, the capturing group always takes part in the match, capturing either "a" or nothing. Backreferences to a capturing group that took part in the match and captured nothing always succeed. Conditionals evaluating such groups execute the "then" part. In short: if you want to use a reference to a group in a conditional, use «(a)?» instead of «(a?)». Continuing with our regex, «b» matches "b". The regex engine now evaluates the conditional. The first capturing group did not take part in the match at all, so the "else" part or «d» is attempted. «d» matches "d" and an overall match is found. 

Moving on to our second subject string "abc", «a» matches "a", which is captured by the capturing group. 

Subsequently, «b» matches "b". The regex engine again evaluates the conditional. The capturing group took part in the match, so the "then" part or «c» is attempted. «c» matches "c" and an overall match is found. Our third subject "bc" does not start with "a", so the capturing group does not take part in the match attempt, like we saw with the first subject string. «b» still matches "b", and the engine moves on to the conditional. The first capturing group did not take part in the match at all, so the "else" part or «d» is attempted. «d» does not match "c" and the match attempt at the start of the string fails. The engine does try again starting at the second character in the string, but fails since «b» does not match "c". 

The fourth subject "abd" is the most interesting one. Like in the second string, the capturing group grabs the "a" and the «b» matches. The capturing group took part in the match, so the "then" part or «c» is attempted. 

«c» fails to match "d", and the match attempt fails. Note that the "else" part is not attempted at this point. 

The capturing group took part in the match, so only the "then" part is used. However, the regex engine isn't done yet. It will restart the regular expression from the beginning, moving ahead one character in the subject string. Starting at the second character in the string, «a» fails to match "b". The capturing group does not take part in the second match attempt which started at the second character in the string. The regex engine moves beyond the optional group, and attempts «b», which matches. The regex engine now arrives at the conditional in the regex, and at the third character in the subject string. The first capturing group did not take part in the current match attempt, so the "else" part or «d» is attempted. «d» matches "d" and an overall match "bd" is found. If you want to avoid this last match result, you need to use anchors. «^(a)?b(?(1)c|d)$» does not find any matches in the last subject string. The caret will fail to match at the second and third characters in the string. 

## Regex Flavors

Conditionals are supported by the JGsoft engine, Perl, PCRE and the .NET framework. All these flavors, except Perl, also support named capturing groups. They allow you to use the name of a capturing group instead of its number as the if test, e.g.: «(?<test>a)?b(?(test)c|d)». Python supports conditionals using a numbered or named capturing group. Python does not support conditionals using lookaround, even though Python does support lookaround outside conditionals. Instead of a conditional like «(?(?=regex)then|else)», you can alternate two opposite lookarounds: 
«(?=regex)then|(?!regex)else)». 

## Example: Extract Email Headers

The regex «^((From|To)|Subject): ((?(2)\w+@\w+\.[a-z]+|.+))» extracts the From, To, and Subject headers from an email message. The name of the header is captured into the first backreference. If the header is the From or To header, it is captured into the second backreference as well. 

The second part of the pattern is the if-then-else conditional «(?(2)\w+@\w+\.[a-z]+|.+))». The if part checks if the second capturing group took part in the match thus far. It will have if the header is the From or To header. In that case, we the *then* part of the conditional «\w+@\w+\.[a-z]+» tries to match an email address. To keep the example simple, we use an overly simple regex to match the email address, and we don't try to match the display name that is usually also part of the From or To header. 

If the second capturing group did not participate in the match this far, the *else* part «.+» is attempted instead. This simply matches the remainder of the line, allowing for any test subject. Finally, we place an extra pair of round brackets around the conditional. This captures the contents of the email header matched by the conditional into the third backreference. The conditional itself does not capture anything. When implementing this regular expression, the first capturing group will store the name of the header ("From", "To", or "Subject"), and the third capturing group will store the value of the header. You could try to match even more headers by putting another conditional into the "else" part. E.g. «^((From|To)|(Date)|Subject): ((?(2)\w+@\w+\.[a-z]+|(?(3)mm/dd/yyyy|.+))» would match a "From", "To", "Date" or "Subject", and use the regex «mm/dd/yyyy» to check if the date is valid. 

Obviously, the date validation regex is just a dummy to keep the example simple. The header is captured in the first group, and its validated contents in the fourth group. As you can see, regular expressions using conditionals quickly become unwieldy. I recommend that you only use them if one regular expression is all your tool allows you to use. When programming, you're far better of using the regex «^(From|To|Date|Subject): (.+)» to capture one header with its unvalidated contents. 

In your source code, check the name of the header returned in the first capturing group, and then use a second regular expression to validate the contents of the header returned in the second capturing group of the first regex. Though you'll have to write a few lines of extra code, this code will be much easier to understand maintain. If you precompile all the regular expressions, using multiple regular expressions will be just as fast, if not faster, and the one big regex stuffed with conditionals. 

## 21. Xml Schema Character Classes

XML Schema Regular Expressions support the usual six shorthand character classes, plus four more. These four aren't supported by any other regular expression flavor. «\i» matches any character that may be the first character of an XML name, i.e. «[_:A-Za-z]». «\c» matches any character that may occur after the first character in an XML name, i.e. «[-._:A-Za-z0-9]». «\I» and «\C» are the respective negated shorthands. Note that the «\c» shorthand syntax conflicts with the control character syntax used in many other regex flavors. 

You can use these four shorthands both inside and outside character classes using the bracket notation. 

They're very useful for validating XML references and values in your XML schemas. The regular expression 
«\i\c*» matches an XML name like "xml:schema". In other regular expression flavors, you'd have to spell this out as «[_:A-Za-z][-._:A-Za-z0-9]*». The latter regex also works with XML's regular expression flavor. It just takes more time to type in. The regex «<\i\c*\s*>» matches an opening XML tag without any attributes. «</\i\c*\s*>» matches any closing tag. «<\i\c*(\s+\i\c*\s*=\s*("[^"]*"|'[^']*'))*\s*>» matches an opening tag with any number of attributes. Putting it all together, «<(\i\c*(\s+\i\c*\s*=\s*("[^"]*"|'[^']*'))*|/\i\c*)\s*>» matches either an opening tag with attributes or a closing tag. 

## Character Class Subtraction

While the regex flavor it defines is quite limited, the XML Schema adds a new regular expression feature not previously seen in any (popular) regular expression flavor: character class subtraction. Currently, this feature is only supported by the JGsoft and .NET regex engines (in addition to those implementing the XML Schema standard). Character class subtraction makes it easy to match any single character present in one list (the character class), but not present in another list (the subtracted class). The syntax for this is [class-[subtract]]. If the character after a hyphen is an opening bracket, XML regular expressions interpret the hyphen as the subtraction operator rather than the range operator. E.g. «[a-z-[aeiuo]]» matches a single letter that is not a vowel (i.e. a single consonant). Without the character class subtraction feature, the only way to do this would be to list all consonants: «[b-df-hj-np-tv-z]». This feature is more than just a notational convenience, though. You can use the full character class syntax within the subtracted character class. E.g. to match all Unicode letters except ASCII letters (i.e. all nonEnglish letters), you could easily use «[\p{L}-[\p{IsBasicLatin}]]». 

## Nested Character Class Subtraction

Since you can use the full character class syntax within the subtracted character class, you can subtract a class from the class being subtracted. E.g. «[0-9-[0-6-[0-3]]]» first subtracts 0-3 from 0-6, yielding «[0-9-
[4-6]]», or «[0-37-9]», which matches any character in the string "0123789". The class subtraction must always be the last element in the character class. [0-9-[4-6]a-f] is not a valid regular expression. It should be rewritten as «[0-9a-f-[4-6]]». The subtraction works on the whole class. E.g. «[\p{Ll}\p{Lu}-[\p{IsBasicLatin}]]» matches all uppercase and lowercase Unicode letters, except any ASCII letters. The \p{IsBasicLatin} is subtracted from the combination of \p{Ll}\p{Lu}
rather than from \p{Lu} alone. This regex will not match "abc". While you can use nested character class subtraction, you cannot subtract two classes sequentially. To subtract ASCII letters and Greek letters from a class with all Unicode letters, combine the ASCII and Greek letters into one class, and subtract that, as in «[\p{L}-[\p{IsBasicLatin}\p{IsGreek}]]». 

## Notational Compatibility With Other Regex Flavors

Note that a regex like «[a-z-[aeiuo]]» will not cause any errors in regex flavors that do not support character class subtraction. But it won't match what you intended either. E.g. in Perl, this regex consists of a character class followed by a literal «]». The character class matches a character that is either in the range a-z, or a hyphen, or an opening bracket, or a vowel. Since the a-z range and the vowels are redundant, you could write this character class as «[a-z-[]» or «[-[a-z]». A hyphen after a range is treated as a literal character, just like a hyphen immediately after the opening bracket. This is true in all regex flavors, including XML. E.g. «[a-z-_]» matches a lowercase letter, a hyphen or an underscore in both Perl and XML Schema. While the last paragraph strictly speaking means that the XML Schema character class syntax is incompatible with Perl and the majority of other regex flavors, in practice there's no difference. Using non-alphanumeric characters in character class ranges is very bad practice, as it relies on the order of characters in the ASCII 
character table, which makes the regular expression hard to understand for the programmer who inherits your work. E.g. while «[A-[]» would match any upper case letter or an opening square bracket in Perl, this regex is much clearer when written as «[A-Z[]». The former regex would cause an error in XML Schema, because it interprets -[] as an empty subtracted class, leaving an unbalanced [. 

## 22. Posix Bracket Expressions

POSIX bracket expressions are a special kind of character classes. POSIX bracket expressions match one character out of a set of characters, just like regular character classes. The main purpose of the bracket expressions is that they adapt to the user's or application's locale. A locale is a collection of rules and settings that describe language and cultural conventions, like sort order, date format, etc. The POSIX standard also defines these locales. Generally, only POSIX-compliant regular expression engines have proper and full support for POSIX bracket expressions. Some non-POSIX regex engines support POSIX character classes, but usually don't support collating sequences and character equivalents. Regular expression engines that support Unicode use Unicode properties and scripts to provide functionality similar to POSIX bracket expressions. In Unicode regex engines, shorthand character classes like «\w» normally match all relevant Unicode characters, alleviating the need to use locales. 

## Character Classes

Don't confuse the POSIX term "character class" with what is normally called a regular expression character class. «[x-z0-9]» is an example of what we call a "character class" and POSIX calls a "bracket expression". 

[:digit:] is a POSIX character class, used inside a bracket expression like «[x-z[:digit:]]». These two regular expressions match exactly the same: a single character that is either "x", "y", "z" or a digit. The class names must be written all lowercase. POSIX bracket expressions can be negated. «[^x-z[:digit:]]» matches a single character that is not x, y, z or a digit. A major difference between POSIX bracket expressions and the character classes in other regex flavors is that POSIX bracket expressions treat the backslash as a literal character. This means you can't use backslashes to escape the closing bracket (]), the caret (^) and the hyphen (-). To include a caret, place it anywhere except right after the opening bracket. «[x^]» matches an x or a caret. You can put the closing bracket right after the opening bracket, or the negating caret. «[]x]» matches a closing bracket or an x. 

«[^]x]» matches any character that is not a closing bracket or an x. The hyphen can be included right after the opening bracket, or right before the closing bracket, or right after the negating caret. Both «[-x]» and «[x-]» match an x or a hyphen. Exactly which POSIX character classes are available depends on the POSIX locale. The following are usually supported, often also by regex engines that don't support POSIX itself. I've also indicated equivalent character classes that you can use in ASCII and Unicode regular expressions if the POSIX classes are unavailable. Some classes also have Perl-style shorthand equivalents. 

Java does not support POSIX bracket expressions, but does support POSIX character classes using the \p operator. Though the \p syntax is borrowed from the syntax for Unicode properties, the POSIX classes in Java only match ASCII characters as indicated below. The class names are case sensitive. Unlike the POSIX syntax which can only be used inside a bracket expression, Java's \p can be used inside and outside bracket expressions. 

| POSIX:            | «[:alnum:]»                                                                |
|-------------------|----------------------------------------------------------------------------|
| Description:      | Alphanumeric characters                                                    |
| ASCII:            | «[a-zA-Z0-9]»                                                              |
| Unicode:          | «[\p{L&}\p{Nd}]»                                                           |
| Shorthand:  Java: | «\p{Alnum}»                                                                |
| POSIX:            | «[:alpha:]»                                                                |
| Description:      | Alphabetic characters                                                      |
| ASCII:            | «[a-zA-Z]»                                                                 |
| Unicode:          | «\p{L&}»                                                                   |
| Shorthand:  Java: | «\p{Alpha}»                                                                |
| POSIX:            | «[:ascii:]»                                                                |
| Description:      | ASCII characters                                                           |
| ASCII:            | «[\x00-\x7F]»                                                              |
| Unicode:          | «\p{InBasicLatin}»                                                         |
| Shorthand:  Java: | «\p{ASCII}»                                                                |
| POSIX:            | «[:blank:]»                                                                |
| Description:      | Space and tab                                                              |
| ASCII:            | «[ \t]»                                                                    |
| Unicode:          | «[\p{Zs}\t]»                                                               |
| Shorthand:  Java: | «\p{Blank}»                                                                |
| POSIX:            | «[:cntrl:]»                                                                |
| Description:      | Control characters                                                         |
| ASCII:            | «[\x00-\x1F\x7F]»                                                          |
| Unicode:          | «\p{Cc}»                                                                   |
| Shorthand:  Java: | «\p{Cntrl}»                                                                |
| POSIX:            | «[:digit:]»                                                                |
| Description:      | Digits                                                                     |
| ASCII:            | «[0-9]»                                                                    |
| Unicode:          | «\p{Nd}»                                                                   |
| Shorthand:        | «\d»                                                                       |
| Java:             | «\p{Digit}»                                                                |
| POSIX:            | «[:graph:]»                                                                |
| Description:      | Visible characters (i.e. anything except spaces, control characters, etc.) |
| ASCII:            | «[\x21-\x7E]»                                                              |
| Unicode:          | «[^\p{Z}\p{C}]»                                                            |
| Shorthand:  Java: | «\p{Graph}»                                                                |

63

| POSIX:            | «[:lower:]»                                                                   |
|-------------------|-------------------------------------------------------------------------------|
| Description:      | Lowercase letters                                                             |
| ASCII:            | «[a-z]»                                                                       |
| Unicode:          | «\p{Ll}»                                                                      |
| Shorthand:  Java: | «\p{Lower}»                                                                   |
| POSIX:            | «[:print:]»                                                                   |
| Description:      | Visible characters and spaces (i.e. anything except control characters, etc.) |
| ASCII:            | «[\x20-\x7E]»                                                                 |
| Unicode:          | «\P{C}»                                                                       |
| Shorthand:  Java: | «\p{Print}»                                                                   |
| POSIX:            | «[:punct:]»                                                                   |
| Description:      | Punctuation characters.                                                       |
| ASCII:            | «[!"#$%&'()*+,-./:;?@[\\\]_`{|}~]»                                            |
| Unicode:          | «\p{P}»                                                                       |
| Shorthand:  Java: | «\p{Punct}»                                                                   |
| POSIX:            | «[:space:]»                                                                   |
| Description:      | All whitespace characters, including line breaks                              |
| ASCII:            | «[ \t\r\n\v\f]»                                                               |
| Unicode:          | «[\p{Z}\t\r\n\v\f]»                                                           |
| Shorthand:        | «\s»                                                                          |
| Java:             | «\p{Space}»                                                                   |
| POSIX:            | «[:upper:]»                                                                   |
| Description:      | Uppercase letters                                                             |
| ASCII:            | «[A-Z]»                                                                       |
| Unicode:          | «\p{Lu}»                                                                      |
| Shorthand:  Java: | «\p{Upper}»                                                                   |
| POSIX:            | «[:word:]»                                                                    |
| Description:      | Word characters (letters, numbers and underscores)                            |
| ASCII:            | «[A-Za-z0-9_]»                                                                |
| Unicode:          | «[\p{L}\p{N}\p{Pc}]»                                                          |
| Shorthand:        | «\w»                                                                          |
| Java:  POSIX:     | «[:xdigit:]»                                                                  |
| Description:      | Hexadecimal digits                                                            |
| ASCII:            | «[A-Fa-f0-9]»                                                                 |
| Unicode:          | «[A-Fa-f0-9]»                                                                 |
| Shorthand:  Java: | «\p{XDigit}»                                                                  |

## Collating Sequences

A POSIX locale can have collating sequences to describe how certain characters or groups of characters should be ordered. E.g. in Spanish, "ll" like in "tortilla" is treated as one character, and is ordered between "l" and "m" in the alphabet. You can use the collating sequence element [.span-ll.] inside a bracket expression to match "ll". E.g. the regex «torti[[.span-ll.]]a» matches "tortilla". Notice the double square brackets. One pair for the bracket expression, and one pair for the collating sequence. 

I do not know of any regular expression engine that support collating sequences, other than POSIXcompliant engines part of a POSIX-compliant system. 

Note that a fully POSIX-compliant regex engine will treat "ll" as a single character when the locale is set to Spanish. This means that «torti[^x]a» also matches "tortilla". «[^x]» matches a single character that is not an "x", which includes "ll" in the Spanish POSIX locale. In any other regular expression engine, or in a POSIX engine not using the Spanish locale, «torti[^x]a» will match the misspelled word "tortila" but will not match "tortilla", as «[^x]» cannot match the two characters "ll". Finally, note that not all regex engines claiming to implement POSIX regular expressions actually have full support for collating sequences. Sometimes, these engines use the regular expression syntax defined by POSIX, but don't have full locale support. You may want to try the above matches to see if the engine you're using does. E.g. Tcl's regexp command supports collating sequences, but Tcl only supports the Unicode locale, which does not define any collating sequences. The result is that in Tcl, a collating sequence specifying a single character will match just that character, and all other collating sequences will result in an error. 

## Character Equivalents

A POSIX locale can define character equivalents that indicate that certain characters should be considered as identical for sorting. E.g. in French, accents are ignored when ordering words. "ÈlËve" comes before 
"Ítre" which comes before "ÈvÈnement". "È" and "Í" are all the same as "e", but "l" comes before "t" which comes before "v". With the locale set to French, a POSIX-compliant regular expression engine will match "e", "È", "Ë" and "Í" when you use the collating sequence [=e=] in the bracket expression 
«[[=e=]]». If a character does not have any equivalents, the character equivalence token simply reverts to the character itself. E.g. «[[=x=][=z=]]» is the same as «[xz]» in the French locale. Like collating sequences, POSIX character equivalents are not available in any regex engine that I know of, other than those following the POSIX standard. And those that do may not have the necessary POSIX locale support. Here too Tcl's regexp command supports character equivalents, but Unicode locale, the only one Tcl supports, does not define any character equivalents. This effectively means that «[[=x=]]» and «[x]» are exactly the same in Tcl, and will only match "x", for any character you may try instead of "x". 

# 23. Adding Comments To Regular Expressions

If you have worked through the entire tutorial, I guess you will agree that regular expressions can quickly become rather cryptic. Therefore, many modern regex flavors allow you to insert comments into regexes. The syntax is «(?\#comment)» where "comment" can be whatever you want, as long as it does not contain a closing round bracket. The regex engine ignores everything after the «(?\#» until the first closing round bracket. E.g. I could clarify the regex to match a valid date by writing it as «(?\#year)(19|20)\d\d[- /.](?\#month)(0[1-9]|1[012])[- /.](?\#day)(0[1-9]|[12][0-9]|3[01])» . Now it is instantly obvious that this regex matches a date in yyyy-mm-dd format. Some software, such as RegexBuddy, EditPad Pro and PowerGREP can apply syntax coloring to regular expressions while you write them. That makes the comments really stand out, enabling the right comment in the right spot to make a complex regular expression much easier to understand. Regex comments are supported by the JGsoft engine, .NET, Perl, PCRE, Python and Ruby. To make your regular expression even more readable, you can turn on free-spacing mode. All flavors that 

![69_image_0.png](69_image_0.png)

support comments also support free-spacing mode. In addition, Java supports free-spacing mode, even though it doesn't support (?\#)-style comments. 

## 24. Free-Spacing Regular Expressions

The JGsoft engine, .NET, Java, Perl, PCRE, Python and Ruby support a variant of the regular expression syntax called free-spacing mode. You can turn on this mode with the «(?x)» mode modifier, or by turning on the corresponding option in the application or passing it to the regex constructor in your programming language. In free-spacing mode, whitespace between regular expression tokens is ignored. Whitespace includes spaces, tabs and line breaks. Note that only whitespace *between* tokens is ignored. E.g. «a b c» is the same as «abc» in free-spacing mode, but «\ d» and «\d» are not the same. The former matches " d", while the latter matches a digit. «\d» is a single regex token composed of a backslash and a "d". Breaking up the token with a space gives you an escaped space (which matches a space), and a literal "d". Likewise, grouping modifiers cannot be broken up. «(?>atomic)» is the same as «(?> ato mic )» and as «( ?>ato mic)». They all match the same atomic group. They're not the same as (? >atomic). In fact, the latter will cause a syntax error. The ?> grouping modifier is a single element in the regex syntax, and must stay together. This is true for all such constructs, including lookaround, named groups, etc. 

A character class is also treated as a single token. «[abc]» is not the same as «[ a b c ]». The former matches one of three letters, while the latter matches those three letters or a space. In other words: freespacing mode has no effect inside character classes. Spaces and line breaks inside character classes will be included in the character class. This means that in free-spacing mode, you can use «\ » or «[ ]» to match a single space. Use whichever you find more readable. 

## Comments In Free-Spacing Mode

Another feature of free-spacing mode is that the \# character starts a comment. The comment runs until the end of the line. Everything from the \# until the next line break character is ignored. Putting it all together, I could clarify the regex to match a valid date by writing it across multiple lines as: 

| # Match a 20th or 21st century date in yyyy-mm-dd format  (19|20)\d\d # year (group 1)  [- /.] # separator  (0[1-9]|1[012]) # month (group 2)  [- /.] # separator  (0[1-9]|[12][0-9]|3[01]) # day (group 3)   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
