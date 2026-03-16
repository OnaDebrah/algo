'use client';

import {useState} from 'react';
import {Popover, PopoverContent, PopoverTrigger,} from '@/components/ui/popover';
import {Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList,} from '@/components/ui/command';
import {Check, ChevronsUpDown} from 'lucide-react';
import countryList from 'react-select-country-list';
import {cn} from "@/utils/formatters";

const CountrySelect = ({
                           value,
                           onChange,
                       }: {
    value: string;
    onChange: (value: string) => void;
}) => {
    const [open, setOpen] = useState(false);
    const countries = countryList().getData();


    const getFlagEmoji = (countryCode: string) => {
        const codePoints = countryCode
            .toUpperCase()
            .split('')
            .map((char) => 127397 + char.charCodeAt(0));
        return String.fromCodePoint(...codePoints);
    };

    const triggerClass = "w-full flex items-center justify-between bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 text-sm text-slate-300 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all cursor-pointer";

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <div
                    role='combobox'
                    aria-expanded={open}
                    className={triggerClass}
                >
                    {value ? (
                        <span className='flex items-center gap-2 overflow-hidden truncate'>
                            <span>{getFlagEmoji(value)}</span>
                            <span className="truncate">{countries.find((c) => c.value === value)?.label}</span>
                        </span>
                    ) : (
                        <span className="text-slate-500">Select...</span>
                    )}
                    <ChevronsUpDown className='ml-2 h-4 w-4 shrink-0 opacity-50 text-slate-400'/>
                </div>
            </PopoverTrigger>
            <PopoverContent
                className='w-(--radix-popover-trigger-width) p-0 bg-slate-900 border-slate-700 shadow-2xl'
                align='start'
            >
                <Command className='bg-slate-900'>
                    <CommandInput
                        placeholder='Search countries...'
                        className='h-10 text-slate-200 placeholder:text-slate-600 border-none focus:ring-0 bg-transparent'
                    />
                    <CommandEmpty className='py-3 text-center text-sm text-slate-500'>
                        No country found.
                    </CommandEmpty>
                    <CommandList className='max-h-60 scrollbar-hide-default border-t border-slate-800'>
                        <CommandGroup>
                            {countries.map((country) => (
                                <CommandItem
                                    key={country.value}
                                    value={`${country.label} ${country.value}`}
                                    onSelect={() => {
                                        onChange(country.value);
                                        setOpen(false);
                                    }}
                                    className='flex items-center px-3 py-2 text-sm text-slate-300 aria-selected:bg-violet-600 aria-selected:text-white cursor-pointer transition-colors'
                                >
                                    <Check
                                        className={cn(
                                            'mr-2 h-4 w-4 text-violet-400',
                                            value === country.value ? 'opacity-100' : 'opacity-0'
                                        )}
                                    />
                                    <span className='flex items-center gap-2'>
                                        <span>{getFlagEmoji(country.value)}</span>
                                        <span>{country.label}</span>
                                    </span>
                                </CommandItem>
                            ))}
                        </CommandGroup>
                    </CommandList>
                </Command>
            </PopoverContent>
        </Popover>
    );
};

export default CountrySelect;
