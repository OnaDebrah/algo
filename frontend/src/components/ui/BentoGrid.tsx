'use client';

import React, {ReactNode} from "react";
import {motion} from "framer-motion";
import {cn} from "@/utils/formatters";

export const BentoGrid = ({
    className,
    children,
}: {
    className?: string;
    children?: ReactNode;
}) => {
    return (
        <div
            className={cn(
                "grid md:auto-rows-[18rem] grid-cols-1 md:grid-cols-3 gap-4 max-w-7xl mx-auto",
                className
            )}
        >
            {children}
        </div>
    );
};

export const BentoGridItem = ({
    className,
    title,
    description,
    header,
    icon,
    onClick,
}: {
    className?: string;
    title?: string | ReactNode;
    description?: string | ReactNode;
    header?: ReactNode;
    icon?: ReactNode;
    onClick?: () => void;
}) => {
    return (
        <motion.div
            whileHover={{ y: -4 }}
            transition={{ duration: 0.2 }}
            onClick={onClick}
            className={cn(
                "row-span-1 border border-slate-800/80 bg-slate-900/50 hover:bg-slate-800/50 rounded-xl group/bento transition duration-200 shadow-xl p-4 justify-between flex flex-col space-y-4 backdrop-blur-sm",
                onClick ? "cursor-pointer" : "",
                className
            )}
        >
            {header}
            <div className="group-hover/bento:translate-x-1 transition duration-200">
                {icon}
                <div className="font-sans font-bold text-slate-200 mt-2">
                    {title}
                </div>
                <div className="font-sans font-normal text-slate-400 text-xs mt-1">
                    {description}
                </div>
            </div>
        </motion.div>
    );
};
